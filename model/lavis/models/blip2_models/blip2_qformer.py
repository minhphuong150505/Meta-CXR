"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from model.lavis.common.registry import registry
from model.lavis.models.base_model import concat_all_gather
from model.lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from model.lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

from mhcac.explanation import ExplanationLoss, explanation_lambda
from mhcac.mhcac_12 import AbnormalityClassificationModel, StreamLayout


from vision_encoders.pubmedclip.pubmed_clip import Pubmedclip
from vision_encoders.swin.swin_encoder import SwinEncoder
from vision_encoders.rad_dino.rad_dino_encoder import RadDinoEncoder
from vision_encoders.stream_adapter import (
    ContrastiveProjectionHead,
    StreamAdapter,
    pool_stream,
)
from vision_encoders.shared_visual_tokens import SharedVisualTokenProjector
# from vision_encoders.medclip.medclip import Medclip

# Common dimension every encoder is projected to before the merge.
VISUAL_DIM = 1408

from mhcac.loss import (
    ClassificationLoss,
    MentionGateLoss,
    MultiPositiveContrastiveLoss,
    MentionConditionedClassificationLoss,
    mention_marginal_log_probs,
    soft_target_kl_loss,
    view_consistency_loss,
)
from mhcac.view_fusion import ViewFusionModule

chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]


def _resolve_encoder_ablation(stream_names, active_encoders):
    """Return trained streams to mask for an inference-only active set."""
    if not active_encoders:
        return ()

    built = tuple(str(name) for name in stream_names)
    active = {str(name) for name in active_encoders}
    unknown = sorted(active - set(built))
    if unknown:
        raise ValueError(
            f"active_encoders names {unknown}, but this model was built "
            f"with {sorted(built)}"
        )
    return tuple(name for name in built if name not in active)


def _hard_negative_sampling_weights(
    similarities: torch.Tensor,
    candidate_valid: torch.Tensor,
    positive_indices: torch.Tensor,
) -> torch.Tensor:
    """Return finite per-row probabilities over valid non-positive samples.

    Masking the positive *after* softmax is numerically unsafe: once the model
    becomes confident, BF16 can assign the positive probability 1 and every
    negative probability 0. Removing the positive then leaves an all-zero row,
    which makes ``torch.multinomial`` trigger a CUDA device-side assertion.

    Compute the distribution in FP32 with positives excluded before softmax.
    A uniform fallback keeps training alive if non-finite similarities still
    erase every learned weight in a row.
    """
    if similarities.ndim != 2:
        raise ValueError("similarities must have shape [batch, candidates]")
    batch_size, candidate_count = similarities.shape
    candidate_valid = candidate_valid.to(
        device=similarities.device, dtype=torch.bool
    ).reshape(-1)
    positive_indices = positive_indices.to(
        device=similarities.device, dtype=torch.long
    ).reshape(-1)
    if candidate_valid.numel() != candidate_count:
        raise ValueError("candidate_valid does not match similarities")
    if positive_indices.numel() != batch_size:
        raise ValueError("positive_indices does not match similarities")

    allowed = candidate_valid.unsqueeze(0).expand(batch_size, -1).clone()
    rows = torch.arange(batch_size, device=similarities.device)
    allowed[rows, positive_indices] = False
    fallback_total = allowed.sum(dim=1, keepdim=True)
    if (fallback_total == 0).any():
        raise ValueError("hard-negative sampling needs at least one candidate per row")

    logits = torch.nan_to_num(
        similarities.detach().float(), nan=-1.0e4, posinf=1.0e4, neginf=-1.0e4
    )
    logits = logits.masked_fill(~allowed, float("-inf"))
    weights = F.softmax(logits, dim=1).masked_fill(~allowed, 0.0)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    totals = weights.sum(dim=1, keepdim=True)
    normalized = weights / totals.clamp_min(torch.finfo(weights.dtype).tiny)
    fallback = allowed.float() / fallback_total.float()
    bad_rows = (~torch.isfinite(totals)) | (totals <= 0)
    return torch.where(bad_rows, fallback, normalized)

@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        use_biovil=True,
        use_pubmedclip=True,
        use_swin=False,
        swin_model_name="ChayanM/SwinV2-GPT2_Mimic",
        swin_backend="hf",
        swin_pretrained=True,
        swin_frozen=True,
        swin_normalize=None,
        use_raddino=False,
        raddino_model_name="microsoft/rad-dino",
        raddino_frozen=True,
        raddino_normalize=True,
        multi_view=False,
        view_fusion_cfg=None,
        lambda_mpc=0.0,
        mpc_warmup_steps=0,
        lambda_view_consistency=0.0,
        view_consistency_cfg=None,
        lambda_itc=1.0,
        lambda_itm=1.0,
        lambda_lm=1.0,
        lambda_cls=1.0,
        lambda_teacher_cls=0.5,
        lambda_distill=0.5,
        lambda_mhcac_contrastive=0.1,
        lambda_orthogonality=0.05,
        lambda_sparsity=0.01,
        lambda_explanation=0.0,
        lambda_explanation_strong=0.0,
        lambda_gate=0.0,
        lambda_mention_conditioned_cls=0.0,
        gate_class_weights=None,
        mention_conditioned_pos_weights=None,
        explanation_cfg=None,
        distill_temperature=2.0,
        mhcac_text_dropout=0.2,
        class_weights=None,
        cls_label_smoothing=0.05,
        uncertain_policy="three_class",
        itc_queue_size=1024,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.use_biovil = bool(use_biovil)
        self.use_pubmedclip = bool(use_pubmedclip)
        self.use_swin = bool(use_swin)
        self.use_raddino = bool(use_raddino)
        if not any([self.use_biovil, self.use_pubmedclip, self.use_swin, self.use_raddino]):
            raise ValueError("At least one encoder must be enabled: biovil, pubmedclip, swin, or raddino")

        self.vit_model = vit_model
        vis_num_feat = 1408
        if self.use_biovil:
            self.visual_encoder, self.ln_vision, vis_num_feat = self.init_vision_encoder(
                "biovil", img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
        else:
            self.visual_encoder = None
            self.ln_vision = nn.LayerNorm(vis_num_feat)

        if freeze_vit and self.visual_encoder is not None:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, vis_num_feat, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(
            len(self.tokenizer), mean_resizing=False
        )
        # transformers>=4.50 replaces the decoder during resize but leaves the
        # custom Q-Former head's separate bias alias at the old vocabulary size.
        # Re-link it so BLIP2's 30,523-token checkpoint loads without a shape
        # mismatch and both state-dict names still reference the same parameter.
        self.Qformer.cls.predictions.bias = self.Qformer.cls.predictions.decoder.bias
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.lambda_itc = float(lambda_itc)
        self.lambda_itm = float(lambda_itm)
        self.lambda_lm = float(lambda_lm)
        self.lambda_cls = float(lambda_cls)
        self.lambda_teacher_cls = float(lambda_teacher_cls)
        self.lambda_distill = float(lambda_distill)
        self.lambda_mhcac_contrastive = float(lambda_mhcac_contrastive)
        self.lambda_orthogonality = float(lambda_orthogonality)
        self.lambda_sparsity = float(lambda_sparsity)
        # Two explanation terms with different evidentiary status:
        #   weak   = pooled CAM vs CheXmask lung (anatomical prior, ~93% of studies)
        #   strong = per-pathology CAM vs MS-CXR expert box (823 train studies)
        # Either being > 0 enables the module; both 0 disables it entirely,
        # including CAM capture.
        self.lambda_explanation = float(lambda_explanation)
        self.lambda_explanation_strong = float(lambda_explanation_strong)
        self.lambda_gate = float(lambda_gate)
        # Hierarchical replacement for (cls_loss + gate BCE). When > 0 it owns
        # the classification objective and both of those are rejected, because
        # optimising the same heads under two disagreeing objectives is how the
        # gate ended up disconnected from the prediction in the first place.
        self.lambda_mention_conditioned_cls = float(lambda_mention_conditioned_cls)
        if self.lambda_mention_conditioned_cls > 0:
            if self.lambda_gate > 0:
                raise ValueError(
                    "lambda_mention_conditioned_cls subsumes lambda_gate; set "
                    "lambda_gate: 0.0"
                )
            if float(lambda_cls) > 0:
                raise ValueError(
                    "lambda_mention_conditioned_cls subsumes lambda_cls; set "
                    "lambda_cls: 0.0"
                )
        self.mention_conditioned_loss_fn = (
            MentionConditionedClassificationLoss(
                num_abnormalities=14,
                pos_weights=mention_conditioned_pos_weights,
            )
            if self.lambda_mention_conditioned_cls > 0
            else None
        )
        self.current_epoch = 0
        explanation_cfg = dict(explanation_cfg or {})
        self.explanation_warmup_start_epoch = int(
            explanation_cfg.get("warmup_start_epoch", 0)
        )
        self.explanation_warmup_epochs = int(
            explanation_cfg.get("warmup_epochs", 0)
        )
        explanation_streams = explanation_cfg.get("streams")
        if isinstance(explanation_streams, str):
            explanation_streams = [explanation_streams]
        self.explanation_streams = (
            None
            if explanation_streams is None
            else tuple(str(name) for name in explanation_streams)
        )
        self.explanation_loss_fn = (
            ExplanationLoss(
                top_k=float(explanation_cfg.get("top_k", 0.5)),
                strong_top_k=explanation_cfg.get("strong_top_k"),
            )
            if (self.lambda_explanation > 0 or self.lambda_explanation_strong > 0)
            else None
        )
        self.distill_temperature = float(distill_temperature)
        self.itc_queue_size = int(itc_queue_size)
        if self.itc_queue_size < 0:
            raise ValueError("itc_queue_size must be non-negative")
        # A detached cross-batch queue makes ITC meaningful for the L4 recipe's
        # microbatch of two.  fp16 keeps a 1,024 x 32 x 256 image queue ~16 MB.
        self.register_buffer(
            "itc_image_queue",
            torch.zeros(
                self.itc_queue_size,
                num_query_token,
                embed_dim,
                dtype=torch.float16,
            ),
            persistent=False,
        )
        self.register_buffer(
            "itc_text_queue",
            torch.zeros(self.itc_queue_size, embed_dim, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "itc_queue_ptr", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "itc_queue_filled", torch.zeros((), dtype=torch.long), persistent=False
        )
        
        # Spatial/intensity augmentation is applied once in ReportDataset so
        # every encoder sees the same mildly transformed radiograph.
        self.pubmedclip = (
            # project=False: SharedVisualTokenProjector owns the 1408 projection.
            Pubmedclip(aug=None, project=False).eval() if self.use_pubmedclip else None
        )

        self.swin = (
            SwinEncoder(
                model_name=swin_model_name,
                pretrained=swin_pretrained,
                frozen=swin_frozen,
                backend=swin_backend,
                normalize=swin_normalize,
            ).eval()
            if self.use_swin
            else None
        )
        swin_dim = self.swin.embed_dim if self.use_swin else 768

        self.raddino = (
            RadDinoEncoder(
                model_name=raddino_model_name,
                pretrained=True,
                frozen=raddino_frozen,
                normalize=raddino_normalize,
            ).eval()
            if self.use_raddino
            else None
        )
        raddino_dim = self.raddino.embed_dim if self.use_raddino else 768

        # self.medclip = Medclip().eval()

        # Multi-view fusion: one module per enabled encoder, operating on that
        # encoder's raw (pre-projection) output, so both the MHCAC branch and the
        # Q-Former concat are built from fused tokens. Zero-init makes this an
        # exact identity at step 0, hence regression-free against a single-view
        # checkpoint.
        self.multi_view = bool(multi_view)
        self.lambda_mpc = float(lambda_mpc)
        # Counted in microbatches, not optimizer updates: this only shapes a
        # loss weight, so the cheaper counter is fine. 0 disables the ramp.
        self.mpc_warmup_steps = int(mpc_warmup_steps)
        self.register_buffer(
            "mpc_step", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.lambda_view_consistency = float(lambda_view_consistency)
        # Soft/conditional agreement knobs. Defaults reproduce the original
        # unconditional symmetric KL, so the previous recipe stays runnable.
        view_consistency_cfg = dict(view_consistency_cfg or {})
        self.view_consistency_margin = float(
            view_consistency_cfg.get("margin", 0.0)
        )
        self.view_consistency_confidence_gate = bool(
            view_consistency_cfg.get("confidence_gate", False)
        )
        self.view_consistency_gate_tolerance = float(
            view_consistency_cfg.get("gate_tolerance", 0.0)
        )
        self.view_fusion = None
        self.mpc_loss_fn = None
        # Pre-fusion streams are only stashed when an auxiliary loss consumes
        # them; otherwise holding the references just wastes memory.
        self._keep_prefusion = bool(multi_view) and (
            self.lambda_mpc > 0 or self.lambda_view_consistency > 0
        )
        if self.multi_view:
            vf_cfg = dict(view_fusion_cfg or {})
            vf_cfg.pop("dim_source", None)
            stream_dims = {}
            if self.use_biovil:
                stream_dims["biovil"] = vis_num_feat
            if self.use_pubmedclip:
                stream_dims["pubmedclip"] = 768
            if self.use_swin:
                stream_dims["swin"] = swin_dim
            if self.use_raddino:
                stream_dims["raddino"] = raddino_dim
            self.view_fusion = nn.ModuleDict({
                name: ViewFusionModule(dim=dim, **vf_cfg)
                for name, dim in stream_dims.items()
            })
            # Trainable capacity between the frozen encoders and the stash
            # point. Without it MultiPositiveContrastiveLoss has no parameter
            # upstream and is a constant -- measured at 3.994 +/- 0.001 for four
            # epochs while carrying 0.1 of the loss weight. Zero-init makes each
            # adapter an exact identity at step 0.
            # ⚠ These are INFERENCE-path parameters: a checkpoint trained
            # without them cannot be resumed into a model that has them.
            self.stream_adapters = nn.ModuleDict({
                name: StreamAdapter(dim=dim) for name, dim in stream_dims.items()
            })
            self.mpc_loss_fn = (
                MultiPositiveContrastiveLoss() if self.lambda_mpc > 0 else None
            )
            # SimCLR-style g(.), training-only, one per stream so the contrastive
            # objective gets its own space instead of pulling on the features
            # MHCAC reads.
            self.mpc_heads = (
                nn.ModuleDict({
                    name: ContrastiveProjectionHead(dim=dim)
                    for name, dim in stream_dims.items()
                })
                if self.lambda_mpc > 0
                else None
            )
            logging.info(f"multi-view fusion enabled for streams: {stream_dims}")

        self._last_raddino_patches = None
        # The single projection/merge point. Both MHCAC and the Q-Former read the
        # tokens it produces, so the two branches can no longer drift onto
        # different visual representations.
        shared_stream_dims = {}
        if self.use_biovil:
            # ln_vision already emits VISUAL_DIM, so this stream gets an Identity.
            shared_stream_dims["biovil"] = VISUAL_DIM
        if self.use_pubmedclip:
            shared_stream_dims["pubmedclip"] = 768
        if self.use_swin:
            shared_stream_dims["swin"] = swin_dim
        if self.use_raddino:
            shared_stream_dims["raddino"] = raddino_dim
        self.shared_visual_projector = SharedVisualTokenProjector(
            shared_stream_dims, visual_dim=VISUAL_DIM
        )
        # Inference-only per-encoder ablation (paper Table 5). Empty tuple is
        # the trained multi-encoder model and leaves the original path intact.
        # Masking happens after shared projection, preserving token positions,
        # expert tokens, MHCAC, and every other learned component.
        self.ablate_encoders: tuple[str, ...] = ()

        self.mhcac = AbnormalityClassificationModel(
            embed_dim=768,
            num_abnormalities=14,
            num_classes=3,
            num_layers=6,
            num_commmon_tokens=14,
            initial_expert_tokens=None,
            visual_dim=VISUAL_DIM,
            text_dropout_rate=mhcac_text_dropout,
            use_cnn=self.use_biovil,
            uncertain_policy=uncertain_policy,
            stream_layouts=self._native_stream_layouts(img_size),
        )

        # sqrt(negative prevalence / class prevalence), capped at 10, computed
        # from the full study-level training cohort (negative, positive, uncertain).
        # [1.0, sqrt(neg/pos), sqrt(neg/uncertain)], capped at 10, measured on
        # the 222,758 study-level train rows of processed/full_allviews_v2.
        #
        # These are far below 1 because blank CheXpert cells are masked rather
        # than counted as negatives (ReportDataset.IGNORE_LABEL). Once "not
        # mentioned" stops inflating the negative class, positives become the
        # majority for 12 of 14 findings -- Atelectasis is 44,718 positive
        # against 1,502 negative -- so the weighting now pushes the other way.
        # Recompute these whenever the manifest or the blank policy changes;
        # the previous values assumed blank == negative and are 3-40x too high
        # for this labelling.
        #
        # No Finding has zero negatives by construction (CheXpert only ever
        # marks it 1 or blank), so it is single-class under this policy and its
        # weight is neutral. It is already excluded from macro metrics by
        # run.include_meta_labels: false.
        default_class_weights = [
            [1.0, 1.00, 0.0],   # No Finding -- single class, see above
            [1.0, 0.86, 0.75],  # Enlarged Cardiomediastinum
            [1.0, 0.60, 1.63],  # Cardiomegaly
            [1.0, 0.24, 0.90],  # Lung Opacity
            [1.0, 0.37, 0.87],  # Lung Lesion
            [1.0, 0.98, 1.40],  # Edema
            [1.0, 0.86, 1.37],  # Consolidation
            [1.0, 1.22, 1.16],  # Pneumonia
            [1.0, 0.18, 0.39],  # Atelectasis
            [1.0, 2.01, 6.11],  # Pneumothorax
            [1.0, 0.71, 2.17],  # Pleural Effusion
            [1.0, 0.26, 0.41],  # Pleural Other
            [1.0, 0.45, 1.27],  # Fracture
            [1.0, 0.23, 3.86],  # Support Devices
        ]

        if class_weights is None:
            class_weights = default_class_weights
        elif len(class_weights) == 0:
            class_weights = None
        self.cls_loss_fn = ClassificationLoss(
            class_weights=class_weights,
            num_abnormalities=14,
            label_smoothing=cls_label_smoothing,
            uncertain_policy=uncertain_policy,
        )
        # The mention gate is what gives the model somewhere to put "nothing to
        # report". Built unconditionally so the parameter set does not depend on
        # a loss weight, but it only receives gradient while lambda_gate > 0.
        self.gate_loss_fn = MentionGateLoss(
            num_abnormalities=14, pos_weights=gate_class_weights
        )
        
    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)


    def _create_mask(self, embeddings, mask_ratio=0.1):
        num_patches = embeddings.size(1)
        num_masked = int(mask_ratio * num_patches)
        
        # Create a mask of ones, then set a subset to zero
        mask = torch.ones(num_patches, device=embeddings.device)
        mask[:num_masked] = 0
        mask = mask[torch.randperm(num_patches)]  # Shuffle to randomize masked positions
        
        # Expand mask to match embeddings' dimensions and apply
        mask = mask.unsqueeze(0).expand(embeddings.size(0), -1)
        mask = mask.unsqueeze(-1)  # Add dimension for broadcasting
        return embeddings * mask  # Apply mask by element-wise multiplication

    def _encode_aux_streams(self, aux_image, cached=None):
        """[B, N, 3, H, W] -> dict[name, [B, N, P, D]] of raw frozen-encoder output.

        Batched (one encoder call over B*N images, not a per-image loop) and
        under no_grad. That detaches the auxiliary *features* only -- the fusion
        module's W_K/W_V are applied outside this block and still get gradient.
        """
        cached = cached or {}
        B, N = aux_image.shape[:2] if aux_image is not None else (0, 0)
        streams = {}

        def unflatten(x):
            return x.reshape(B, N, *x.shape[1:])

        # Cached auxiliary features arrive already shaped [B, N, P, D].
        for name in ("biovil", "pubmedclip", "swin", "raddino"):
            if name in cached:
                streams[name] = cached[name]

        need = [
            name for name, enabled in (
                ("biovil", self.use_biovil), ("pubmedclip", self.use_pubmedclip),
                ("swin", self.use_swin), ("raddino", self.use_raddino),
            ) if enabled and name not in streams
        ]
        if not need:
            return streams
        if aux_image is None:
            raise ValueError(
                f"aux_image is required to encode auxiliary streams {need}; "
                "the requested encoders are not covered by the feature cache."
            )

        flat = aux_image.flatten(0, 1)
        with torch.no_grad():
            if "biovil" in need:
                streams["biovil"] = unflatten(
                    self.visual_encoder(flat).projected_patch_embeddings.reshape(
                        flat.shape[0], -1, 1408
                    )
                )
            if "pubmedclip" in need:
                # Fuse the 768-dim ViT stream; the 1408 projection is recomputed
                # from the fused tokens, so the aux projection is discarded here.
                streams["pubmedclip"] = unflatten(
                    self.pubmedclip(flat, apply_aug=False)[0]
                )
            if "swin" in need:
                streams["swin"] = unflatten(self.swin(flat))
            if "raddino" in need:
                streams["raddino"] = unflatten(self.raddino(flat))
        return streams

    def _stash_prefusion(self, name, anchor, aux_streams):
        """Keep the pre-fusion tensors the auxiliary losses need, if any do."""
        if self._keep_prefusion:
            self._last_prefusion_streams[name] = (anchor, aux_streams.get(name))

    def _mpc_lambda(self):
        """Linear ramp for the contrastive weight over the first epoch.

        The head is randomly initialised, so at step 0 this term is pure noise
        pulling on the adapter that everything else also reads. Ramping it in is
        cheap insurance; the previous 0.1 was only harmless because the term had
        no gradient at all.
        """
        if self.lambda_mpc <= 0 or self.mpc_warmup_steps <= 0:
            return self.lambda_mpc
        if self.training:
            self.mpc_step += 1
        progress = float(self.mpc_step.item()) / float(self.mpc_warmup_steps)
        return self.lambda_mpc * min(1.0, progress)

    def _adapt(self, name, tokens):
        """Residual adapter for one stream; identity when fusion is disabled."""
        adapters = getattr(self, "stream_adapters", None)
        if adapters is None or name not in adapters:
            return tokens
        return adapters[name](tokens)

    def _fuse(self, name, anchor, aux_streams, aux_mask, anchor_view_id, aux_view_ids):
        """Fuse one encoder stream with its auxiliary views, if multi-view is on."""
        if not self.multi_view or self.view_fusion is None:
            return anchor
        aux = aux_streams.get(name)
        if aux is None:
            return anchor
        return self.view_fusion[name](
            anchor, aux.to(anchor.dtype), aux_mask, anchor_view_id, aux_view_ids
        )

    def _native_stream_layouts(self, img_size):
        """Per-encoder token layout, or ``None`` to use the legacy resample path.

        Returning a layout tells MHCAC to keep every encoder's own sequence
        intact. That is the point of running two encoders: BioViL-T reads the
        full 448 image into a 14x14 grid (32 px per cell, fine detail),
        PubMedCLIP reads it at its native 224 into a 7x7 grid (64 px per cell,
        regional context) plus one CLS token for the global view. Pooling both
        down to 7x7 -- what this used to do -- collapsed three scales into two
        copies of the coarsest one.

        ``None`` is returned whenever an enabled encoder cannot be described,
        because a partial layout would leave that stream without a positional
        encoding. Swin and RadDINO do not expose their token count here, so
        those recipes keep the historical behaviour unchanged.
        """
        if self.use_swin or self.use_raddino:
            return None
        layouts = {}
        if self.use_biovil:
            # BioViL-T's ResNet50 trunk has total stride 32: 448 -> 14x14 = 196.
            # MHCAC re-checks this against the actual span and raises if it is
            # wrong, so a future change to the trunk fails loudly here.
            #
            # img_size must be the size the dataset actually emits. It is NOT
            # self-evidently so: init_vision_encoder ignores it for biovil, so
            # for a long time nothing read this value and the BLIP-2 default of
            # 224 went unnoticed while the vis_processor produced 448. The run
            # YAML now sets model.image_size explicitly for that reason.
            if img_size is None:
                raise ValueError(
                    "model.image_size is required to lay out the biovil stream; "
                    "set it to the vis_processor image_size (448)"
                )
            grid = int(img_size) // 32
            layouts["biovil"] = StreamLayout(grid * grid)
        if self.use_pubmedclip:
            vision_cfg = self.pubmedclip.model.config.vision_config
            grid = vision_cfg.image_size // vision_cfg.patch_size
            layouts["pubmedclip"] = StreamLayout(
                grid * grid + 1, num_global_tokens=1
            )
        return layouts or None

    def _apply_encoder_ablation(self, shared):
        """Mask configured encoder spans while preserving the trained layout."""
        if not self.ablate_encoders:
            return shared
        if self.training:
            raise RuntimeError("active_encoders is inference-only; call model.eval()")
        return shared.without(*self.ablate_encoders)

    def _encode_image_streams(self, image, apply_aug=False, cached=None,
                              aux_image=None, aux_cached=None, aux_mask=None,
                              anchor_view_id=None, aux_view_ids=None):
        # ``cached`` holds raw frozen-encoder outputs (before ln_vision /
        # *_qformer_proj) precomputed by pretraining/precompute_features.py. When
        # present we skip the frozen encoder forward; the trainable projection
        # layers below still run so training is identical.
        cached = cached or {}
        # Raw (pre-merge) per-encoder outputs, keyed by stream name. The shared
        # projector is the only thing that turns these into visual tokens.
        raw_streams = {}
        self._last_raddino_patches = None
        self._last_prefusion_streams = {}

        # Multi-view: each stream is fused at its raw, pre-projection output, so
        # the trainable projections below run once on the fused [B, P, D] tensor
        # and both the MHCAC and Q-Former branches see fused tokens.
        # A batch where no study has an auxiliary view (N_max == 0) skips the
        # auxiliary encode entirely rather than running encoders on empty input.
        has_aux_input = (
            aux_image is not None and aux_image.shape[1] > 0
        ) or any(v is not None and v.shape[1] > 0 for v in (aux_cached or {}).values())
        fuse_on = self.multi_view and self.view_fusion is not None
        aux_streams = (
            self._encode_aux_streams(aux_image, cached=aux_cached)
            if fuse_on and has_aux_input
            else {}
        )
        # The encoder forward above runs under torch.no_grad(); the adapter must
        # not. Applying it here is what lets the auxiliary side carry gradient.
        aux_streams = {
            name: self._adapt(name, tokens)
            for name, tokens in aux_streams.items()
        }

        if self.use_biovil:
            cnn_raw = cached["biovil"] if "biovil" in cached else (
                self.visual_encoder(image).projected_patch_embeddings.reshape(
                    image.shape[0], -1, VISUAL_DIM
                )
            )
            cnn_raw = self._adapt("biovil", cnn_raw)
            self._stash_prefusion("biovil", cnn_raw, aux_streams)
            cnn_raw = self._fuse("biovil", cnn_raw, aux_streams, aux_mask,
                                 anchor_view_id, aux_view_ids)
            # ln_vision normalises the frozen encoder output; it is not a
            # dimension projection, so it stays on the encoder side of the merge.
            raw_streams["biovil"] = self.ln_vision(cnn_raw)

        if self.use_pubmedclip:
            if "pubmedclip" in cached:
                vit_patches = cached["pubmedclip"]
            else:
                vit_patches, _ = self.pubmedclip(image, apply_aug=apply_aug)
            vit_patches = self._adapt("pubmedclip", vit_patches)
            self._stash_prefusion("pubmedclip", vit_patches, aux_streams)
            vit_patches = self._fuse("pubmedclip", vit_patches, aux_streams,
                                     aux_mask, anchor_view_id, aux_view_ids)
            # The encoder's own ``mlp`` head is no longer used to reach VISUAL_DIM:
            # the shared projector owns that projection for every stream alike.
            raw_streams["pubmedclip"] = vit_patches

        if self.use_swin:
            swin_patches = cached["swin"] if "swin" in cached else self.swin(image)
            swin_patches = self._adapt("swin", swin_patches)
            self._stash_prefusion("swin", swin_patches, aux_streams)
            raw_streams["swin"] = self._fuse(
                "swin", swin_patches, aux_streams, aux_mask, anchor_view_id, aux_view_ids
            )

        if self.use_raddino:
            raddino_patches = cached["raddino"] if "raddino" in cached else self.raddino(image)
            raddino_patches = self._adapt("raddino", raddino_patches)
            self._stash_prefusion("raddino", raddino_patches, aux_streams)
            raddino_patches = self._fuse("raddino", raddino_patches, aux_streams,
                                         aux_mask, anchor_view_id, aux_view_ids)
            self._last_raddino_patches = raddino_patches
            raw_streams["raddino"] = raddino_patches

        if not raw_streams:
            raise ValueError("No image encoder stream is enabled.")

        # One merge, one representation, consumed by both downstream branches.
        shared = self.shared_visual_projector(raw_streams)
        return self._apply_encoder_ablation(shared)
    
    def initialize_expert_tokens(self, chexpert_cols, embed_dim):
        # Initialize expert tokens based on text embeddings of abnormality names
        expert_embeddings = []
        for abnormality in chexpert_cols:
            # Get the text embedding (CLS token) for each abnormality
            text_tokens = self.tokenizer(
                abnormality,
                padding="max_length",
                truncation=True,
                max_length=20,  # Adjust max_length if necessary
                return_tensors="pt",
            ).to(next(self.parameters()).device)  # Move to device of the model
            
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            cls_embedding = text_output.last_hidden_state[:, 0, :]  # CLS token embedding
            expert_embeddings.append(cls_embedding)

        # Stack embeddings and return as initialized expert tokens
        embeddings = torch.cat(expert_embeddings, dim=0).reshape(len(chexpert_cols), embed_dim)
        torch.save(embeddings, "weights/expert_embeddings.pt")
        return embeddings
        
    
    @staticmethod
    def _batch_mask(samples, key, batch_size, device, fallback_key=None, default=True):
        value = samples.get(key)
        if value is None and fallback_key is not None:
            value = samples.get(fallback_key)
        if value is None:
            return torch.full((batch_size,), default, dtype=torch.bool, device=device)
        mask = torch.as_tensor(value, dtype=torch.bool, device=device)
        if mask.ndim == 0:
            mask = mask.expand(batch_size)
        mask = mask.reshape(-1)
        if mask.numel() != batch_size:
            raise ValueError(f"{key} must contain one boolean per batch item")
        return mask

    @staticmethod
    def _gather_with_local_grad(tensor):
        """Gather equal-sized DDP batches while preserving this rank's gradient.

        Remote features are constants on each rank; DDP later combines the
        parameter gradients produced where each feature was local.  This also
        avoids a backward collective deadlock when one rank has no valid report.
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        gathered = concat_all_gather(tensor.detach())
        batch_size = tensor.shape[0]
        start = dist.get_rank() * batch_size
        return torch.cat(
            [gathered[:start], tensor, gathered[start + batch_size :]], dim=0
        )

    @torch.no_grad()
    def _update_itc_queue(self, image_features, text_features, valid_mask):
        if self.itc_queue_size == 0 or not self.training:
            return
        images = concat_all_gather(image_features.detach())
        texts = concat_all_gather(text_features.detach())
        valid = concat_all_gather(valid_mask).bool()
        images = images[valid]
        texts = texts[valid]
        if images.shape[0] == 0:
            return
        if images.shape[0] >= self.itc_queue_size:
            images = images[-self.itc_queue_size :]
            texts = texts[-self.itc_queue_size :]

        count = images.shape[0]
        pointer = int(self.itc_queue_ptr.item())
        first = min(count, self.itc_queue_size - pointer)
        self.itc_image_queue[pointer : pointer + first].copy_(
            images[:first].to(self.itc_image_queue.dtype)
        )
        self.itc_text_queue[pointer : pointer + first].copy_(
            texts[:first].to(self.itc_text_queue.dtype)
        )
        remaining = count - first
        if remaining:
            self.itc_image_queue[:remaining].copy_(
                images[first:].to(self.itc_image_queue.dtype)
            )
            self.itc_text_queue[:remaining].copy_(
                texts[first:].to(self.itc_text_queue.dtype)
            )
        self.itc_queue_ptr.fill_((pointer + count) % self.itc_queue_size)
        self.itc_queue_filled.fill_(
            min(self.itc_queue_size, int(self.itc_queue_filled.item()) + count)
        )

    def _image_text_contrastive(self, image_features, text_features, valid_mask):
        image_features_all = self._gather_with_local_grad(image_features)
        text_features_all = self._gather_with_local_grad(text_features)
        valid_all = concat_all_gather(valid_mask)

        current_count = valid_all.numel()
        # The queue is a training-only source of negatives. Validation must not
        # depend on whichever train samples happened to fill the ring buffer at
        # the end of an epoch.
        queue_filled = int(self.itc_queue_filled.item()) if self.training else 0
        if queue_filled:
            # clone() decouples autograd's saved operands from the ring-buffer
            # update performed later in this same forward pass.
            image_features_all = torch.cat(
                [
                    image_features_all,
                    self.itc_image_queue[:queue_filled]
                    .to(image_features_all.dtype)
                    .clone(),
                ],
                dim=0,
            )
            text_features_all = torch.cat(
                [
                    text_features_all,
                    self.itc_text_queue[:queue_filled]
                    .to(text_features_all.dtype)
                    .clone(),
                ],
                dim=0,
            )
            candidate_valid = torch.cat(
                [
                    valid_all,
                    torch.ones(queue_filled, dtype=torch.bool, device=valid_all.device),
                ]
            )
        else:
            candidate_valid = valid_all

        temperature = self.temp.clamp(min=1e-3, max=0.5)
        sim_i2t = torch.einsum(
            "bqd,nd->bnq", image_features, text_features_all
        ).amax(dim=-1) / temperature
        sim_t2i = torch.einsum(
            "bd,nqd->bnq", text_features, image_features_all
        ).amax(dim=-1) / temperature
        sim_i2t = sim_i2t.masked_fill(
            ~candidate_valid.unsqueeze(0), float("-inf")
        )
        sim_t2i = sim_t2i.masked_fill(
            ~candidate_valid.unsqueeze(0), float("-inf")
        )

        zero = image_features.sum() * 0.0
        if not valid_mask.any():
            return (
                zero,
                sim_i2t[:, :current_count],
                sim_t2i[:, :current_count],
                valid_all,
            )
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        targets = rank * image_features.shape[0] + torch.arange(
            image_features.shape[0], device=image_features.device
        )
        loss_itc = 0.5 * (
            F.cross_entropy(sim_i2t[valid_mask], targets[valid_mask])
            + F.cross_entropy(sim_t2i[valid_mask], targets[valid_mask])
        )
        # ITM needs raw images/token ids, which the lightweight ITC queue does
        # not retain, so hard-negative mining uses the current global batch.
        return (
            loss_itc,
            sim_i2t[:, :current_count],
            sim_t2i[:, :current_count],
            valid_all,
        )

    def _image_text_matching(
        self,
        image_embeds,
        text_tokens,
        valid_mask,
        valid_all,
        sim_i2t,
        sim_t2i,
    ):
        zero = image_embeds.sum() * 0.0
        if not valid_mask.any() or valid_all.sum() < 2:
            return zero

        image_embeds_all = self._gather_with_local_grad(image_embeds)
        text_ids_all_ranks = concat_all_gather(text_tokens.input_ids)
        text_atts_all_ranks = concat_all_gather(text_tokens.attention_mask)
        batch_size = image_embeds.shape[0]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        positive_indices = rank * batch_size + torch.arange(
            batch_size, device=image_embeds.device
        )

        with torch.no_grad():
            weights_t2i = _hard_negative_sampling_weights(
                sim_t2i, valid_all, positive_indices
            )
            weights_i2t = _hard_negative_sampling_weights(
                sim_i2t, valid_all, positive_indices
            )

        local_indices = valid_mask.nonzero(as_tuple=True)[0]
        negative_image_indices = torch.stack(
            [torch.multinomial(weights_t2i[i], 1).squeeze(0) for i in local_indices]
        )
        negative_text_indices = torch.stack(
            [torch.multinomial(weights_i2t[i], 1).squeeze(0) for i in local_indices]
        )

        image_pos = image_embeds[local_indices]
        image_neg = image_embeds_all[negative_image_indices]
        text_ids_pos = text_tokens.input_ids[local_indices]
        text_atts_pos = text_tokens.attention_mask[local_indices]
        text_ids_neg = text_ids_all_ranks[negative_text_indices]
        text_atts_neg = text_atts_all_ranks[negative_text_indices]

        # positive, negative-image, negative-text triples
        text_ids = torch.cat([text_ids_pos, text_ids_pos, text_ids_neg], dim=0)
        text_atts = torch.cat([text_atts_pos, text_atts_pos, text_atts_neg], dim=0)
        image_inputs = torch.cat([image_pos, image_neg, image_pos], dim=0)
        query_tokens = self.query_tokens.expand(text_ids.shape[0], -1, -1)
        query_atts = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=image_embeds.device
        )
        output = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=torch.cat([query_atts, text_atts], dim=1),
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=torch.ones(
                image_inputs.shape[:-1], dtype=torch.long, device=image_embeds.device
            ),
            return_dict=True,
        )
        vl_embeddings = output.last_hidden_state[:, : query_tokens.shape[1]]
        logits = self.itm_head(vl_embeddings).mean(dim=1)
        num_positive = local_indices.numel()
        labels = torch.cat(
            [
                torch.ones(num_positive, dtype=torch.long, device=image_embeds.device),
                torch.zeros(2 * num_positive, dtype=torch.long, device=image_embeds.device),
            ]
        )
        return F.cross_entropy(logits, labels)

    def _language_modeling(self, text_tokens, query_tokens, query_output, valid_mask):
        zero = query_output.last_hidden_state.sum() * 0.0
        if not valid_mask.any():
            return zero

        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )
        labels[~valid_mask] = -100
        query_atts = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=query_tokens.device
        )
        output = self.Qformer(
            decoder_input_ids,
            attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
            reduction="none",
        )
        token_count = (labels[:, 1:] != -100).sum()
        if token_count == 0:
            return zero
        return output.loss.sum() / token_count

    def forward(self, samples):
        image = samples.get("image")
        text = samples["text_output"]
        cached = {
            k: samples[f"{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin", "raddino")
            if f"{k}_feat" in samples
        }
        aux_cached = {
            k: samples[f"aux_{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin", "raddino")
            if f"aux_{k}_feat" in samples
        }
        shared_visual = self._encode_image_streams(
            image,
            apply_aug=False,
            cached=cached,
            aux_image=samples.get("aux_image"),
            aux_cached=aux_cached,
            aux_mask=samples.get("aux_mask"),
            anchor_view_id=samples.get("anchor_view_id"),
            aux_view_ids=samples.get("aux_view_ids"),
        )
        image_embeds = shared_visual.tokens
        device = image_embeds.device
        batch_size = image_embeds.shape[0]
        classification_mask = self._batch_mask(
            samples,
            "classification_mask",
            batch_size,
            device,
            fallback_key="has_chexpert_label",
        )
        generation_mask = self._batch_mask(
            samples, "generation_mask", batch_size, device
        )

        # The vision-language objectives and the privileged-text teacher are the
        # only consumers of the Q-Former and text-encoder forwards.  Running them
        # when every weight that reads them is zero is pure waste: two BERT
        # forwards, ITM hard-negative mining and an LM decoder pass, all
        # multiplied by zero afterwards.  A classification-only recipe therefore
        # skips the block entirely rather than paying for it.
        #
        # Consequence, by design: with the vision-language weights at zero the
        # Q-Former receives no gradient and stays at its pretrained
        # initialisation, so the checkpoint cannot serve the Stage-2 soft-token
        # modes.  See CLAUDE.md.
        needs_vision_language = (
            self.lambda_itc > 0 or self.lambda_itm > 0 or self.lambda_lm > 0
        )
        needs_text_encoder = needs_vision_language or (
            self.lambda_teacher_cls > 0 or self.lambda_distill > 0
        )

        zero = image_embeds.sum() * 0.0
        loss_itc, loss_itm, loss_lm = zero, zero, zero
        text_tokens = None
        text_output = None

        if needs_text_encoder:
            text_tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )

        if needs_vision_language:
            image_atts = torch.ones(
                image_embeds.shape[:-1], dtype=torch.long, device=device
            )
            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                use_cache=True,
                return_dict=True,
            )
            image_features = F.normalize(
                self.vision_proj(query_output.last_hidden_state), dim=-1
            )
            text_features = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0]), dim=-1
            )

            # ITM reuses ITC's similarity matrices for hard-negative sampling, so
            # the contrastive pass still runs when only ITM is enabled.
            if self.lambda_itc > 0 or self.lambda_itm > 0:
                loss_itc, sim_i2t, sim_t2i, generation_mask_all = (
                    self._image_text_contrastive(
                        image_features, text_features, generation_mask
                    )
                )
                if self.lambda_itm > 0:
                    loss_itm = self._image_text_matching(
                        image_embeds,
                        text_tokens,
                        generation_mask,
                        generation_mask_all,
                        sim_i2t,
                        sim_t2i,
                    )
            if self.lambda_lm > 0:
                loss_lm = self._language_modeling(
                    text_tokens, query_tokens, query_output, generation_mask
                )
            if self.lambda_itc > 0:
                self._update_itc_queue(
                    image_features, text_features, generation_mask
                )

        cls_labels = samples["classification_labels"]
        lambda_eff = 0.0
        explanation_mask = samples.get("explanation_mask")
        explanation_valid = None
        capture_explanation = False
        if self.explanation_loss_fn is not None:
            lambda_eff = explanation_lambda(
                self.current_epoch,
                max(self.lambda_explanation, self.lambda_explanation_strong),
                self.explanation_warmup_start_epoch,
                self.explanation_warmup_epochs,
            )
            mask_valid = samples.get("explanation_mask_valid")
            # Grad-CAM needs a live graph: it differentiates the score with
            # respect to the visual activations.  RunnerBase.eval_epoch is
            # decorated @torch.no_grad(), and validation reaches this same
            # forward(), so without this guard the first scored epoch dies with
            # "element 0 of tensors does not require grad".  Skipping the term
            # under no_grad is also what we want on merit -- an explanation
            # penalty that trains nothing has no business costing a backward.
            if (
                lambda_eff > 0
                and torch.is_grad_enabled()
                and explanation_mask is not None
                and mask_valid is not None
            ):
                explanation_valid = self._batch_mask(
                    samples,
                    "explanation_mask_valid",
                    batch_size,
                    device,
                    default=False,
                ) & classification_mask
                # The WEAK term must see CheXmask lungs only. The pooled cache
                # stores whichever annotation the builder preferred, and
                # `choose_preferred_mask` prefers the MS-CXR bbox when a study
                # has one -- so without this filter the weak term would run on a
                # bbox union for exactly the 869 train / 164 test studies that
                # also feed the strong term, supervising them twice and making
                # "weak = anatomical prior" false where it matters most.
                # mask_source: 0 = CheXmask lung, 1 = MS-CXR bbox.
                mask_source = samples.get("explanation_mask_source")
                if mask_source is not None:
                    is_lung = torch.as_tensor(
                        mask_source, device=device
                    ).reshape(-1) == 0
                    if is_lung.numel() != batch_size:
                        raise ValueError(
                            "explanation_mask_source must hold one value per "
                            "batch item"
                        )
                    weak_valid = explanation_valid & is_lung
                else:
                    weak_valid = explanation_valid
                positive_study = (cls_labels.to(device=device) == 1).any(dim=1)
                capture_explanation = bool(
                    (explanation_valid & positive_study).any().item()
                )

        cam_streams = None
        self.mhcac.capture_streams = capture_explanation
        try:
            student_logits, _, contrastive_loss, orth_loss, sparsity_loss, mention_logits = self.mhcac(
                shared_visual,
                text_embeddings=None,
                labels=cls_labels,
                sample_mask=classification_mask,
            )
        finally:
            if capture_explanation:
                cam_streams = self.mhcac._last_cam_streams
            self.mhcac.capture_streams = False
            self.mhcac._last_cam_streams = None

        loss_explanation = None
        if self.explanation_loss_fn is not None:
            loss_explanation = student_logits.sum() * 0.0
            if capture_explanation and cam_streams:
                selected_streams = {
                    name: value
                    for name, value in cam_streams.items()
                    if self.explanation_streams is None
                    or name in self.explanation_streams
                }
                (
                    loss_explanation_weak,
                    loss_explanation_strong,
                    _,
                ) = self.explanation_loss_fn(
                    student_logits,
                    cls_labels,
                    selected_streams,
                    explanation_mask,
                    weak_valid,
                    # Skipped outright when the strong weight is zero, rather
                    # than computed and multiplied by zero: each distinct boxed
                    # finding in the batch costs its own autograd.grad.
                    bbox_masks=(
                        samples.get("explanation_bbox_masks")
                        if self.lambda_explanation_strong > 0
                        else None
                    ),
                    bbox_valid=(
                        samples.get("explanation_bbox_valid")
                        if self.lambda_explanation_strong > 0
                        else None
                    ),
                )
                # Ratio against the shared warmup so each term keeps its own
                # weight while the schedule shape stays common.
                peak = max(
                    self.lambda_explanation, self.lambda_explanation_strong
                )
                loss_explanation = (
                    (self.lambda_explanation / peak) * loss_explanation_weak
                    + (self.lambda_explanation_strong / peak)
                    * loss_explanation_strong
                )
        # Hierarchy, when enabled: the gate multiplies into the classifier and
        # `student_logits` becomes the LOG MARGINAL distribution, so everything
        # downstream (argmax, softmax, the saved .npz) keeps working unchanged
        # while silence can finally veto a positive.
        loss_mention_conditioned = student_logits.sum() * 0.0
        if self.mention_conditioned_loss_fn is not None:
            mention_targets = samples.get("mention_targets")
            if mention_targets is None:
                raise ValueError(
                    "lambda_mention_conditioned_cls > 0 but the batch carries no "
                    "mention_targets; the dataset must emit them"
                )
            # MUST be mention_mask: that is the key ReportDataset emits. Asking
            # for has_chexpert_label silently fell through to default=True and
            # trained every unmatched study as fourteen "not mentioned" cells.
            mention_mask = self._batch_mask(
                samples, "mention_mask", batch_size, device, default=True
            )
            loss_mention_conditioned = self.mention_conditioned_loss_fn(
                student_logits,
                mention_logits,
                cls_labels,
                mention_targets.to(mention_logits.device),
                sample_mask=mention_mask,
            )
            # Keep `student_logits` = q, the polarity distribution CONDITIONAL on
            # the finding being mentioned. The CheXpert P/N/U metric masks blank
            # cells, so that is exactly the quantity it scores.
            #
            # The four-state joint is exported alongside, never in place of it:
            #   P(blank) = 1 - m,  P(Neg) = m*q_neg,  P(Pos) = m*q_pos,
            #   P(Unc) = m*q_unc
            # Substituting the three-state marginal here (which aliased blank
            # onto Negative) made Positive unwinnable under argmax and pinned
            # validation F1 at exactly 0.000 for every epoch of the smoke.
            # Report-time emission is a two-stage decision -- open the gate on a
            # per-label threshold fitted on validation, then read the class off
            # q -- not an argmax over a marginal.
            marginal_log_probs = mention_marginal_log_probs(
                student_logits, mention_logits
            )
        else:
            marginal_log_probs = None

        cls_loss = self.cls_loss_fn(
            student_logits, cls_labels, sample_mask=classification_mask
        ) if self.lambda_cls > 0 else student_logits.sum() * 0.0

        # The teacher may read only a valid FINDINGS target.  The image-only
        # student remains the sole classification path exported at inference.
        teacher_mask = classification_mask & generation_mask
        loss_teacher_cls = student_logits.sum() * 0.0
        loss_distill = student_logits.sum() * 0.0
        if teacher_mask.any() and (
            self.lambda_teacher_cls > 0 or self.lambda_distill > 0
        ):
            teacher_logits, _, _, _, _, _ = self.mhcac(
                shared_visual,
                text_embeddings=text_output.last_hidden_state,
                text_attention_mask=text_tokens.attention_mask,
                # Teacher supervision is applied by ClassificationLoss below;
                # do not duplicate the O(B^2) token contrastive calculation.
                labels=None,
            )
            loss_teacher_cls = self.cls_loss_fn(
                teacher_logits, cls_labels, sample_mask=teacher_mask
            )
            loss_distill = soft_target_kl_loss(
                student_logits,
                teacher_logits,
                sample_mask=teacher_mask,
                temperature=self.distill_temperature,
            )

        loss_mpc = cls_loss.new_zeros(())
        loss_view_consistency = cls_loss.new_zeros(())
        aux_mask = samples.get("aux_mask")
        if self.multi_view and aux_mask is not None and aux_mask.any():
            if self.mpc_loss_fn is not None:
                # Pool per stream, then project. PubMedCLIP has a real CLS
                # token; BioViL has none and its own global output IS the patch
                # mean. Pooling the 246 concatenated tokens instead would weight
                # BioViL 196/246 against PubMedCLIP 50/246 by token count alone,
                # across two different feature spaces.
                terms = []
                for name, (anchor_raw, aux_raw) in (
                    self._last_prefusion_streams.items()
                ):
                    if aux_raw is None or name not in self.mpc_heads:
                        continue
                    head = self.mpc_heads[name]
                    has_cls = name == "pubmedclip"
                    terms.append(
                        self.mpc_loss_fn(
                            head(pool_stream(anchor_raw, has_cls)),
                            head(pool_stream(aux_raw.to(anchor_raw.dtype), has_cls)),
                            aux_mask,
                        )
                    )
                if terms:
                    loss_mpc = torch.stack(terms).mean()
            if self.lambda_view_consistency > 0:
                pre = self._last_prefusion_streams
                anchor_raw_streams = {
                    name: (
                        self.ln_vision(pre[name][0]) if name == "biovil" else pre[name][0]
                    )
                    for name in self.shared_visual_projector.stream_names
                    if name in pre
                }
                anchor_shared = self.shared_visual_projector(anchor_raw_streams)
                anchor_logits, _, _, _, _, _ = self.mhcac(
                    anchor_shared,
                    text_embeddings=None,
                    labels=None,
                )
                loss_view_consistency = view_consistency_loss(
                    student_logits,
                    anchor_logits,
                    aux_mask.any(dim=1),
                    margin=self.view_consistency_margin,
                    confidence_gate=self.view_consistency_confidence_gate,
                    gate_tolerance=self.view_consistency_gate_tolerance,
                )

        total_loss = (
            self.lambda_itc * loss_itc
            + self.lambda_itm * loss_itm
            + self.lambda_lm * loss_lm
            + self.lambda_cls * cls_loss
            + self.lambda_teacher_cls * loss_teacher_cls
            + self.lambda_distill * loss_distill
            + self.lambda_mhcac_contrastive * contrastive_loss
            + self.lambda_orthogonality * orth_loss
            + self.lambda_sparsity * sparsity_loss
            + self._mpc_lambda() * loss_mpc
            + self.lambda_view_consistency * loss_view_consistency
            + self.lambda_mention_conditioned_cls * loss_mention_conditioned
        )
        loss_gate = student_logits.sum() * 0.0
        if self.lambda_gate > 0:
            mention_targets = samples.get("mention_targets")
            if mention_targets is None:
                raise ValueError(
                    "lambda_gate > 0 but the batch carries no mention_targets; "
                    "the dataset must be rebuilt (ReportDataset emits them)"
                )
            loss_gate = self.gate_loss_fn(
                mention_logits,
                mention_targets.to(mention_logits.device),
                sample_mask=self._batch_mask(
                    samples, "mention_mask", batch_size, device, default=True
                ),
            )
            total_loss = total_loss + self.lambda_gate * loss_gate
        if lambda_eff > 0:
            total_loss = total_loss + lambda_eff * loss_explanation
        return BlipOutput(
            loss=total_loss,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_cls=cls_loss,
            loss_teacher_cls=loss_teacher_cls,
            loss_distill=loss_distill,
            loss_contrastive=contrastive_loss,
            loss_orthagonal=orth_loss,
            loss_sparsity=sparsity_loss,
            loss_explanation=loss_explanation,
            loss_gate=loss_gate,
            loss_mpc=loss_mpc,
            loss_view_consistency=loss_view_consistency,
            classification_logits=student_logits,
            mention_marginal_log_probs=marginal_log_probs,
            classification_mask=classification_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples.get("image")
        cached = {
            k: samples[f"{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin", "raddino")
            if f"{k}_feat" in samples
        }
        aux_cached = {
            k: samples[f"aux_{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin", "raddino")
            if f"aux_{k}_feat" in samples
        }
        image_embeds = self._encode_image_streams(
            image,
            apply_aug=False,
            cached=cached,
            aux_image=samples.get("aux_image"),
            aux_cached=aux_cached,
            aux_mask=samples.get("aux_mask"),
            anchor_view_id=samples.get("anchor_view_id"),
            aux_view_ids=samples.get("aux_view_ids"),
        )
        batch_size = image_embeds.shape[0]
        device = image_embeds.device

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(batch_size, 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image, aux_image=None, aux_mask=None,
                      anchor_view_id=None, aux_view_ids=None):
        """Return image-only classification logits and learned Q-Former tokens.

        ``image`` may be a tensor (legacy API) or a complete samples dict.  The
        dict form supports the frozen-feature cache and study auxiliary view.
        Report text is intentionally ignored: this is the student/inference path.
        """
        cached = {}
        aux_cached = {}
        if isinstance(image, dict):
            samples = image
            image = samples.get("image")
            aux_image = samples.get("aux_image", aux_image)
            aux_mask = samples.get("aux_mask", aux_mask)
            anchor_view_id = samples.get("anchor_view_id", anchor_view_id)
            aux_view_ids = samples.get("aux_view_ids", aux_view_ids)
            cached = {
                k: samples[f"{k}_feat"]
                for k in ("biovil", "pubmedclip", "swin", "raddino")
                if f"{k}_feat" in samples
            }
            aux_cached = {
                k: samples[f"aux_{k}_feat"]
                for k in ("biovil", "pubmedclip", "swin", "raddino")
                if f"aux_{k}_feat" in samples
            }
        shared_visual = self._encode_image_streams(
            image, apply_aug=False, cached=cached,
            aux_image=aux_image, aux_mask=aux_mask,
            aux_cached=aux_cached,
            anchor_view_id=anchor_view_id, aux_view_ids=aux_view_ids,
        )

        concat_image_embeds = shared_visual.tokens

        classification_logits, attention, contrastive_loss, orth_loss, sparsity_loss, _ = self.mhcac(
            shared_visual,
            text_embeddings=None,
            labels=None,
        )

        image_atts = torch.ones(
            concat_image_embeds.shape[:-1],
            dtype=torch.long,
            device=concat_image_embeds.device,
        )

        query_tokens = self.query_tokens.expand(concat_image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=concat_image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    
        # print("CNN patches shape:", image_embeds.shape)
        # print("VIT patches shape:", image_embeds_2.shape)
        # image_patches = self.image_embed_proj_norm(self.image_embed_proj(image_embeds))
        # txt_cls_token = text_output.last_hidden_state[:, 0, :]
        # print(f"query_output.last_hidden_state shape: {query_output.last_hidden_state.shape}")

        return classification_logits, query_output.last_hidden_state


    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_output")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self._encode_image_streams(image, apply_aug=False).tokens
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self._encode_image_streams(image, apply_aug=False).tokens
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        vit_model_cls = cfg.get("vit_model_cls", [])
        if isinstance(vit_model_cls, str):
            vit_model_cls = [vit_model_cls]
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)
        encoders = cfg.get("encoders", {}) or {}

        def cfg_bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        use_biovil = cfg_bool(
            encoders.get("biovil", cfg.get("use_biovil", vit_model == "biovil"))
        )
        use_pubmedclip = cfg_bool(
            encoders.get(
                "pubmedclip",
                cfg.get("use_pubmedclip", "pubmedclip" in list(vit_model_cls)),
            )
        )
        use_swin = cfg_bool(encoders.get("swin", cfg.get("use_swin", False)))
        swin_cfg = cfg.get("swin", {}) or {}
        swin_model_name = cfg.get(
            "swin_model_name",
            swin_cfg.get("model_name", "ChayanM/SwinV2-GPT2_Mimic"),
        )
        swin_backend = cfg.get("swin_backend", swin_cfg.get("backend", "hf"))
        swin_pretrained = cfg_bool(
            cfg.get("swin_pretrained", swin_cfg.get("pretrained", True)),
            default=True,
        )
        swin_frozen = cfg_bool(
            cfg.get("swin_frozen", swin_cfg.get("frozen", True)),
            default=True,
        )
        swin_normalize_raw = cfg.get("swin_normalize", swin_cfg.get("normalize", None))
        swin_normalize = (
            None
            if swin_normalize_raw is None
            else cfg_bool(swin_normalize_raw, default=True)
        )

        use_raddino = cfg_bool(encoders.get("raddino", cfg.get("use_raddino", False)))
        raddino_cfg = cfg.get("raddino", {}) or {}
        raddino_model_name = cfg.get(
            "raddino_model_name",
            raddino_cfg.get("model_name", "microsoft/rad-dino"),
        )
        raddino_frozen = cfg_bool(
            cfg.get("raddino_frozen", raddino_cfg.get("frozen", True)),
            default=True,
        )
        raddino_normalize = cfg_bool(
            cfg.get("raddino_normalize", raddino_cfg.get("normalize", True)),
            default=True,
        )

        # Multi-view. from_config silently drops unknown config blocks, so these
        # keys must be read explicitly to take effect.
        multi_view = cfg_bool(cfg.get("multi_view", False))
        view_fusion_cfg_raw = cfg.get("view_fusion", {}) or {}
        view_fusion_cfg = {
            "num_heads": int(view_fusion_cfg_raw.get("num_heads", 8)),
            "ffn_ratio": int(view_fusion_cfg_raw.get("ffn_ratio", 4)),
            "num_blocks": int(view_fusion_cfg_raw.get("num_blocks", 1)),
            "num_view_types": int(view_fusion_cfg_raw.get("num_view_types", 4)),
            "dropout": float(view_fusion_cfg_raw.get("dropout", 0.1)),
            "p_view_drop": float(view_fusion_cfg_raw.get("p_view_drop", 0.2)),
        }
        loss_cfg = cfg.get("loss", {}) or {}
        lambda_mpc = float(loss_cfg.get("lambda_mpc", 0.0))
        mpc_warmup_steps = int(loss_cfg.get("mpc_warmup_steps", 0))
        lambda_view_consistency = float(loss_cfg.get("lambda_view_consistency", 0.0))
        view_consistency_cfg = cfg.get("view_consistency", {}) or {}
        lambda_itc = float(loss_cfg.get("lambda_itc", 1.0))
        lambda_itm = float(loss_cfg.get("lambda_itm", 1.0))
        lambda_lm = float(loss_cfg.get("lambda_lm", 1.0))
        lambda_cls = float(loss_cfg.get("lambda_cls", 1.0))
        lambda_teacher_cls = float(loss_cfg.get("lambda_teacher_cls", 0.5))
        lambda_distill = float(loss_cfg.get("lambda_distill", 0.5))
        lambda_mhcac_contrastive = float(
            loss_cfg.get("lambda_mhcac_contrastive", 0.1)
        )
        lambda_orthogonality = float(loss_cfg.get("lambda_orthogonality", 0.05))
        lambda_sparsity = float(loss_cfg.get("lambda_sparsity", 0.01))
        lambda_explanation = float(loss_cfg.get("lambda_explanation", 0.0))
        lambda_explanation_strong = float(
            loss_cfg.get("lambda_explanation_strong", 0.0)
        )
        lambda_gate = float(loss_cfg.get("lambda_gate", 0.0))
        lambda_mention_conditioned_cls = float(
            loss_cfg.get("lambda_mention_conditioned_cls", 0.0)
        )
        itc_queue_size = int(loss_cfg.get("itc_queue_size", 1024))

        explanation_cfg_raw = cfg.get("explanation", {}) or {}
        explanation_streams = explanation_cfg_raw.get("streams", None)
        if isinstance(explanation_streams, str):
            explanation_streams = [explanation_streams]
        explanation_cfg = {
            "top_k": float(explanation_cfg_raw.get("top_k", 0.5)),
            "warmup_start_epoch": int(
                explanation_cfg_raw.get("warmup_start_epoch", 0)
            ),
            "warmup_epochs": int(explanation_cfg_raw.get("warmup_epochs", 0)),
            "streams": (
                None
                if explanation_streams is None
                else list(explanation_streams)
            ),
        }

        mhcac_cfg = cfg.get("mhcac", {}) or {}
        gate_class_weights = mhcac_cfg.get("gate_class_weights", None)
        mention_conditioned_pos_weights = mhcac_cfg.get(
            "mention_conditioned_pos_weights", None
        )
        distill_temperature = float(mhcac_cfg.get("distill_temperature", 2.0))
        mhcac_text_dropout = float(mhcac_cfg.get("text_dropout", 0.2))
        class_weights = mhcac_cfg.get("class_weights", None)
        if class_weights is not None:
            class_weights = [list(weights) for weights in class_weights]
        cls_label_smoothing = float(mhcac_cfg.get("label_smoothing", 0.05))
        uncertain_policy = str(mhcac_cfg.get("uncertain_policy", "three_class"))

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            use_biovil=use_biovil,
            use_pubmedclip=use_pubmedclip,
            use_swin=use_swin,
            swin_model_name=swin_model_name,
            swin_backend=swin_backend,
            swin_pretrained=swin_pretrained,
            swin_frozen=swin_frozen,
            swin_normalize=swin_normalize,
            use_raddino=use_raddino,
            raddino_model_name=raddino_model_name,
            raddino_frozen=raddino_frozen,
            raddino_normalize=raddino_normalize,
            multi_view=multi_view,
            view_fusion_cfg=view_fusion_cfg,
            lambda_mpc=lambda_mpc,
            mpc_warmup_steps=mpc_warmup_steps,
            lambda_view_consistency=lambda_view_consistency,
            view_consistency_cfg=view_consistency_cfg,
            lambda_itc=lambda_itc,
            lambda_itm=lambda_itm,
            lambda_lm=lambda_lm,
            lambda_cls=lambda_cls,
            lambda_teacher_cls=lambda_teacher_cls,
            lambda_distill=lambda_distill,
            lambda_mhcac_contrastive=lambda_mhcac_contrastive,
            lambda_orthogonality=lambda_orthogonality,
            lambda_sparsity=lambda_sparsity,
            lambda_explanation=lambda_explanation,
            lambda_explanation_strong=lambda_explanation_strong,
            lambda_gate=lambda_gate,
            lambda_mention_conditioned_cls=lambda_mention_conditioned_cls,
            gate_class_weights=gate_class_weights,
            mention_conditioned_pos_weights=mention_conditioned_pos_weights,
            explanation_cfg=explanation_cfg,
            distill_temperature=distill_temperature,
            mhcac_text_dropout=mhcac_text_dropout,
            class_weights=class_weights,
            cls_label_smoothing=cls_label_smoothing,
            uncertain_policy=uncertain_policy,
            itc_queue_size=itc_queue_size,
        )

        # Optional inference-only ablation. ``active_encoders`` names streams
        # to keep; all other streams built into the checkpoint are zeroed at
        # the shared-token boundary. Absent/empty means no ablation.
        model.ablate_encoders = _resolve_encoder_ablation(
            model.shared_visual_projector.stream_names,
            cfg.get("active_encoders", None),
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
