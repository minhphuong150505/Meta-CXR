"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from time import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

from model.lavis.common.registry import registry
from model.lavis.models.base_model import all_gather_with_grad, concat_all_gather
from model.lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from model.lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures

from mhcac.mhcac_12 import AbnormalityClassificationModel


from vision_encoders.pubmedclip.pubmed_clip import Pubmedclip
from vision_encoders.swin.swin_encoder import SwinEncoder
from vision_encoders.rad_dino.rad_dino_encoder import RadDinoEncoder
# from vision_encoders.medclip.medclip import Medclip

from mhcac.utils import compute_metrics_for_tasks
from mhcac.loss import ClassificationLoss, MultiPositiveContrastiveLoss, view_consistency_loss
from mhcac.aggregator import Aggregator
from mhcac.view_fusion import ViewFusionModule

chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

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
        lambda_view_consistency=0.0,
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
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
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
        
        self.vis_augs = Compose([transforms.RandomAffine(degrees=30, shear=15),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2)])
        
        self.vis_transforms = Compose([Resize((224, 224)),
                                        ToTensor()])
        
        self.vit_projection = nn.Linear(768, 1408)

        self.pubmedclip = Pubmedclip(aug=self.vis_augs).eval() if self.use_pubmedclip else None

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
        self.swin_qformer_proj = nn.Linear(swin_dim, 1408) if self.use_swin else None

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
        self.raddino_qformer_proj = nn.Linear(raddino_dim, 1408) if self.use_raddino else None

        # self.medclip = Medclip().eval()

        # Multi-view fusion: one module per enabled encoder, operating on that
        # encoder's raw (pre-projection) output, so both the MHCAC branch and the
        # Q-Former concat are built from fused tokens. Zero-init makes this an
        # exact identity at step 0, hence regression-free against a single-view
        # checkpoint.
        self.multi_view = bool(multi_view)
        self.lambda_mpc = float(lambda_mpc)
        self.lambda_view_consistency = float(lambda_view_consistency)
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
            self.mpc_loss_fn = (
                MultiPositiveContrastiveLoss() if self.lambda_mpc > 0 else None
            )
            logging.info(f"multi-view fusion enabled for streams: {stream_dims}")

        self._last_raddino_patches = None
        self.mhcac = AbnormalityClassificationModel(
            embed_dim=768,
            num_abnormalities=14,
            num_classes=3,
            num_layers=6,
            num_commmon_tokens=14,
            initial_expert_tokens=None,
            swin_dim=swin_dim,
            raddino_dim=raddino_dim,
        )
        
        class_weights = [
            torch.tensor([1.0, 1.0, 0.000], dtype=torch.float),  # Class weights for no finding
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float),  # Class weights for Enlarged Cardiomediastinum
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  # Class weights for Cardiomegaly
            torch.tensor([1.0, 4.0, 10.0], dtype=torch.float),  # Class weights for Lung Opacity
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  # Class weights for Lung Lesion
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),   # Edema
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  # Consolidation
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float),   # Class weights for Pneumonia
            torch.tensor([1.0, 4.0, 10.0], dtype=torch.float),   # Class weights for Atelectasis 
            torch.tensor([1.0, 5.0, 10.0], dtype=torch.float),  #  Pneumothorax 
            torch.tensor([1.0, 4.0, 10.0], dtype=torch.float),  # Class weights for Pleural Effusion 
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float), # Class weights for Pleural Other 
            torch.tensor([1.0, 10.0, 10.0], dtype=torch.float),  # Fracture 
            torch.tensor([1.0, 3.0, 0.0], dtype=torch.float)  # Class weights for Support Devices
        ]  #negative(0),positive(1),uncertain(2)
        
        # class_weights = [
        #     torch.tensor([10.0, 10.0], dtype=torch.float),  # Class weights for no finding
        #     torch.tensor([1.0, 10.0], dtype=torch.float),  # Class weights for Enlarged Cardiomediastinum
        #     torch.tensor([2.0, 10.0], dtype=torch.float),  # Class weights for Cardiomegaly
        #     torch.tensor([4.0, 10.0], dtype=torch.float),  # Class weights for Lung Opacity
        #     torch.tensor([4.0, 10.0], dtype=torch.float),  # Class weights for Lung Lesion
        #     torch.tensor([1.0, 10.0], dtype=torch.float),   # Edema
        #     torch.tensor([1.0, 10.0], dtype=torch.float),  # Consolidation
        #     torch.tensor([1.0, 10.0], dtype=torch.float),   # Class weights for Pneumonia
        #     torch.tensor([5.0, 10.0], dtype=torch.float),   # Class weights for Atelectasis 
        #     torch.tensor([1.0, 10.0], dtype=torch.float),  #  Pneumothorax 
        #     torch.tensor([2.0, 10.0], dtype=torch.float),  # Class weights for Pleural Effusion 
        #     torch.tensor([5.0, 10.0], dtype=torch.float), # Class weights for Pleural Other 
        #     torch.tensor([2.0, 10.0], dtype=torch.float),  # Fracture 
        #     torch.tensor([5.0, 10.0], dtype=torch.float)  # Class weights for Support Devices
        # ]
        
        # print(f"class weights are {class_weights}")

        """
        chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]
                             
        """
        # Instantiate the loss function with abnormality-specific class weights
        self.cls_loss_fn = ClassificationLoss(penalty_weight=0.1, class_weights=class_weights, num_abnormalities=14)  #negative(0),positive(1),uncertain(2)
        

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
        for name in ("biovil", "swin", "raddino"):
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
            # The feature cache only covers biovil/swin/raddino, so an enabled
            # pubmedclip still needs the raw images -- same constraint the
            # single-view path already has.
            raise ValueError(
                f"aux_image is required to encode auxiliary streams {need}; "
                "these encoders are not covered by the feature cache."
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

    def _encode_image_streams(self, image, apply_aug=False, cached=None,
                              aux_image=None, aux_cached=None, aux_mask=None,
                              anchor_view_id=None, aux_view_ids=None):
        # ``cached`` holds raw frozen-encoder outputs (before ln_vision /
        # *_qformer_proj) precomputed by pretraining/precompute_features.py. When
        # present we skip the frozen encoder forward; the trainable projection
        # layers below still run so training is identical.
        cached = cached or {}
        cnn_patches = None
        vit_patches = None
        swin_patches = None
        raddino_patches = None
        qformer_streams = []
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

        if self.use_biovil:
            cnn_raw = cached["biovil"] if "biovil" in cached else (
                self.visual_encoder(image).projected_patch_embeddings.reshape(
                    image.shape[0], -1, 1408
                )
            )
            self._stash_prefusion("biovil", cnn_raw, aux_streams)
            cnn_raw = self._fuse("biovil", cnn_raw, aux_streams, aux_mask,
                                 anchor_view_id, aux_view_ids)
            cnn_patches = self.ln_vision(cnn_raw)
            qformer_streams.append(cnn_patches)

        if self.use_pubmedclip:
            vit_patches, pubmed_projection = self.pubmedclip(image, apply_aug=apply_aug)
            self._stash_prefusion("pubmedclip", vit_patches, aux_streams)
            if fuse_on and "pubmedclip" in aux_streams:
                vit_patches = self._fuse("pubmedclip", vit_patches, aux_streams,
                                         aux_mask, anchor_view_id, aux_view_ids)
                # Recompute the 1408 Q-Former projection from the fused tokens.
                # nn.Sequential of Linear broadcasts over [B, P, 768].
                pubmed_projection = self.pubmedclip.mlp(vit_patches)
            qformer_streams.append(pubmed_projection)

        if self.use_swin:
            swin_patches = cached["swin"] if "swin" in cached else self.swin(image)
            self._stash_prefusion("swin", swin_patches, aux_streams)
            swin_patches = self._fuse("swin", swin_patches, aux_streams, aux_mask,
                                      anchor_view_id, aux_view_ids)
            qformer_streams.append(self.swin_qformer_proj(swin_patches))

        if self.use_raddino:
            raddino_patches = cached["raddino"] if "raddino" in cached else self.raddino(image)
            self._stash_prefusion("raddino", raddino_patches, aux_streams)
            raddino_patches = self._fuse("raddino", raddino_patches, aux_streams,
                                         aux_mask, anchor_view_id, aux_view_ids)
            self._last_raddino_patches = raddino_patches
            qformer_streams.append(self.raddino_qformer_proj(raddino_patches))

        if not qformer_streams:
            raise ValueError("No image encoder stream is enabled.")

        qformer_image_embeds = torch.cat(qformer_streams, dim=1)
        return cnn_patches, vit_patches, swin_patches, qformer_image_embeds
    
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
        
    
    def forward(self, samples):
        start_time = time()
        image = samples.get("image")
        text = samples["text_output"]

        # Use precomputed frozen-encoder features when the dataset provides them
        # (run.feature_cache_dir); otherwise run the encoders on the image.
        cached = {
            k: samples[f"{k}_feat"]
            for k in ("biovil", "swin", "raddino")
            if f"{k}_feat" in samples
        }
        aux_cached = {
            k: samples[f"aux_{k}_feat"]
            for k in ("biovil", "swin", "raddino")
            if f"aux_{k}_feat" in samples
        }
        cnn_patches, vit_patches, swin_patches, qformer_image_embeds = self._encode_image_streams(
            image, apply_aug=False, cached=cached,
            aux_image=samples.get("aux_image"),
            aux_cached=aux_cached,
            aux_mask=samples.get("aux_mask"),
            anchor_view_id=samples.get("anchor_view_id"),
            aux_view_ids=samples.get("aux_view_ids"),
        )
        device = qformer_image_embeds.device

        # image_embeds_3 = self.ln_vision(self.medclip(image))
        # image_embeds_3 = self._create_mask(image_embeds_3)  # Mask 20% of patches


        image_atts = torch.ones(qformer_image_embeds.size()[:-1], dtype=torch.long).to(
            device
        )
        
        # query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        

        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_atts,
        #     use_cache=True,
        #     return_dict=True,
        # )

        # cls_image_feat = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        
        # image_feats = F.normalize(
        #     self.vision_proj(query_output.last_hidden_state), dim=-1
        # )

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
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )
        
        # image_patches = self.image_embed_proj_norm(self.image_embed_proj(image_embeds))
        # txt_cls_token = self.text_cls_proj_norm(self.text_cls_proj(text_output.last_hidden_state[:, 0, :]))
        
        # image_patches_norm =  F.normalize(image_patches, dim=-1)
        # txt_cls_token_norm =  F.normalize(txt_cls_token, dim=-1)
        
        # Hook to log gradient norms to a file
        def log_grad_to_file(grad, filename="gradient_log.txt"):
            with open(filename, 'a') as f:
                f.write(f"Expert token gradient norm: {grad.norm().item()}\n")

        # Register the hook
        self.mhcac.expert_tokens.register_hook(lambda grad: log_grad_to_file(grad))
        
        ### ---- classification loss ----###
        
        cls_labels = samples["classification_labels"]  # Ground truth labels for abnormalities
        classification_logits, attention, contrastive_loss, orth_loss, sparsity_loss = self.mhcac(
            cnn_patches=cnn_patches,
            vit_patches=vit_patches,
            swin_patches=swin_patches,
            raddino_patches=self._last_raddino_patches,
            text_embeddings=text_output.last_hidden_state,
            labels=cls_labels,
        )

        # classification_logits,vit_attention, cnn_attention = self.mhcac(cnn_patches = image_embeds, vit_patches = image_embeds_2, labels = None)  # Output from your MHCAC module

        # Compute abnormality-specific loss
        cls_loss = self.cls_loss_fn(classification_logits, cls_labels)

        ### ---- multi-view auxiliary losses (inert while their lambdas are 0) ----###
        loss_mpc = cls_loss.new_zeros(())
        loss_view_consistency = cls_loss.new_zeros(())
        aux_mask = samples.get("aux_mask")
        if self.multi_view and aux_mask is not None and aux_mask.any():
            if self.mpc_loss_fn is not None:
                terms = [
                    self.mpc_loss_fn(anchor_raw, aux_raw, aux_mask)
                    for anchor_raw, aux_raw in self._last_prefusion_streams.values()
                    if aux_raw is not None
                ]
                if terms:
                    loss_mpc = torch.stack(terms).mean()

            if self.lambda_view_consistency > 0:
                # Second MHCAC pass on the un-fused anchor. This doubles the
                # MHCAC forward cost, which is the expensive trainable part.
                pre = self._last_prefusion_streams
                anchor_logits, _, _, _, _ = self.mhcac(
                    cnn_patches=self.ln_vision(pre["biovil"][0]) if "biovil" in pre else None,
                    vit_patches=pre["pubmedclip"][0] if "pubmedclip" in pre else None,
                    swin_patches=pre["swin"][0] if "swin" in pre else None,
                    raddino_patches=pre["raddino"][0] if "raddino" in pre else None,
                    text_embeddings=text_output.last_hidden_state,
                    labels=cls_labels,
                )
                loss_view_consistency = view_consistency_loss(
                    classification_logits, anchor_logits, aux_mask.any(dim=1)
                )

        metrics = compute_metrics_for_tasks(classification_logits, cls_labels)
        
         ###============== Image-text Contrastive for Claasifcation ===================###
        # sim_q2t = torch.matmul(
        #     image_patches_norm.unsqueeze(1), txt_cls_token_norm.unsqueeze(-1)
        # ).squeeze()

        # # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp

        # # text-query similarity
        # sim_t2q = torch.matmul(
        #     txt_cls_token_norm.unsqueeze(1).unsqueeze(1), image_patches_norm.permute(0, 2, 1)
        # ).squeeze()

        # # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp

        # bs = image.size(0)
        # targets = torch.arange(bs, dtype=torch.long).to(image.device)

        # loss_itc = (
        #                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                #    ) / 2
        
        
        ###============== Image-text Contrastive ===================###
        
        # # Compute the similarity between CLS image and CLS text tokens
        # sim_q2t = torch.matmul(cls_image_feat, cls_text_feat.T)  # [batch_size, batch_size]

        # # Apply temperature scaling
        # sim_q2t = sim_q2t / self.temp

        # # sim_i2t represents the similarity from image to text, sim_t2i from text to image
        # # These matrices should be symmetrical, but we'll compute both for clarity
        # sim_i2t = sim_q2t  # Image-to-Text similarity
        # sim_t2i = sim_q2t.T  # Text-to-Image similarity

        # # Create target indices for positive pairs
        # bs = image.size(0)
        # targets = torch.arange(bs, dtype=torch.long).to(image.device)

        # # Calculate Image-Text Contrastive Loss (ITC)
        # loss_itc = (
        #     F.cross_entropy(sim_i2t, targets)  # Image-to-Text contrastive loss
        #     + F.cross_entropy(sim_t2i, targets)  # Text-to-Image contrastive loss
        # ) / 2
        
        ###============== Image-text Contrastive ===================###
        # sim_q2t = torch.matmul(
        #     image_feats.unsqueeze(1), text_feat.unsqueeze(-1)
        # ).squeeze()

        # # image-text similarity: aggregate across all query tokens
        # sim_i2t, _ = sim_q2t.max(-1)
        # sim_i2t = sim_i2t / self.temp

        # # text-query similarity
        # sim_t2q = torch.matmul(
        #     text_feat.unsqueeze(1).unsqueeze(1), image_feats.permute(0, 2, 1)
        # ).squeeze()

        # # text-image similarity: aggregate across all query tokens
        # sim_t2i, _ = sim_t2q.max(-1)
        # sim_t2i = sim_t2i / self.temp

        # bs = image.size(0)
        # targets = torch.arange(bs, dtype=torch.long).to(image.device)

        # loss_itc = (
        #                    F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        #                    + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        #            ) / 2
        
        ###============== Image-text Matching ===================###
        # with torch.no_grad():
        #     weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
        #     weights_t2i.fill_diagonal_(0)
        #     weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
        #     weights_i2t.fill_diagonal_(0)

        # # select a negative image for each text
        # image_embeds_neg = []
        # for b in range(bs):
        #     clamped_weight = torch.clamp(weights_t2i[b], min=1e-6)
        #     neg_idx = torch.multinomial(clamped_weight, 1).item()
        #     image_embeds_neg.append(image_embeds[neg_idx])
        # image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # # select a negative text for each image
        # text_ids_neg = []
        # text_atts_neg = []
        # for b in range(bs):
        #     clamped_weight = torch.clamp(weights_i2t[b], min=1e-6)
        #     neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        #     text_ids_neg.append(text_tokens.input_ids[neg_idx])
        #     text_atts_neg.append(text_tokens.attention_mask[neg_idx])

        # text_ids_neg = torch.stack(text_ids_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # text_ids_all = torch.cat(
        #     [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        # )  # pos, pos, neg
        # text_atts_all = torch.cat(
        #     [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        #     dim=0,
        # )

        # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        # query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        # image_embeds_all = torch.cat(
        #     [image_embeds, image_embeds_neg, image_embeds], dim=0
        # )  # pos, neg, pos
        # image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )

        # output_itm = self.Qformer.bert(
        #     text_ids_all,
        #     query_embeds=query_tokens_itm,
        #     attention_mask=attention_mask_all,
        #     encoder_hidden_states=image_embeds_all,
        #     encoder_attention_mask=image_atts_all,
        #     return_dict=True,
        # )

        # vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        # vl_output = self.itm_head(vl_embeddings)
        # logits = vl_output.mean(dim=1)

        # itm_labels = torch.cat(
        #     [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        #     dim=0,
        # ).to(image.device)
        # loss_itm = F.cross_entropy(logits, itm_labels)

        # ##================= Image Captioning ========================##
        # decoder_input_ids = text_tokens.input_ids.clone()
        # decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # labels = decoder_input_ids.masked_fill(
        #     decoder_input_ids == self.tokenizer.pad_token_id, -100
        # )

        # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        #     image.device
        # )
        # attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        # lm_output = self.Qformer(
        #     decoder_input_ids,
        #     attention_mask=attention_mask,
        #     past_key_values=query_output.past_key_values,
        #     return_dict=True,
        #     labels=labels,
        # )

        # loss_lm = lm_output.loss
        # # print(self.tokenizer.decode(torch.argmax(lm_output.logits, dim=-1)[0]))
        # loss_lm = 0.0
        end_time = time()
        # print(f"forward function took {end_time - start_time:.4f} seconds")
        
        return BlipOutput(
            loss = cls_loss + contrastive_loss * 0.3 + orth_loss * 0.7 + sparsity_loss * 0.3
                   + self.lambda_mpc * loss_mpc
                   + self.lambda_view_consistency * loss_view_consistency,
            # loss = cls_loss,
            loss_cls=cls_loss,
            loss_contrastive = contrastive_loss,
            loss_orthagonal = orth_loss,
            loss_sparsity = sparsity_loss,
            loss_mpc = loss_mpc,
            loss_view_consistency = loss_view_consistency,
            average_precision = metrics['average']['precision'],
            average_recall = metrics['average']['recall'],
            average_accuracy = metrics['average']['accuracy'],
            average_f1_score = metrics['average']['f1_score'],
        )
        
        # return BlipOutput(
        #     loss=loss_itm + loss_itc*1.5 + loss_lm,
        #     loss_itc=loss_itc*1.5,
        #     loss_itm=loss_itm,
        #     loss_lm=loss_lm,
        # )

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
        image = samples["image"].cuda()
        _, _, _, image_embeds = self._encode_image_streams(image, apply_aug=False)

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
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
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image, aux_image=None, aux_mask=None,
                      anchor_view_id=None, aux_view_ids=None):
        cnn_patches, vit_patches, swin_patches, concat_image_embeds = self._encode_image_streams(
            image, apply_aug=False,
            aux_image=aux_image, aux_mask=aux_mask,
            anchor_view_id=anchor_view_id, aux_view_ids=aux_view_ids,
        )

        classification_logits, attention, contrastive_loss, orth_loss, sparsity_loss = self.mhcac(
            cnn_patches=cnn_patches,
            vit_patches=vit_patches,
            swin_patches=swin_patches,
            raddino_patches=self._last_raddino_patches,
            text_embeddings=None,
            labels=None,
        )

        image_atts = torch.ones(concat_image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(concat_image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=concat_image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,
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
                _, _, _, image_embeds_frozen = self._encode_image_streams(image, apply_aug=False)
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
                _, _, _, image_embeds_frozen = self._encode_image_streams(image, apply_aug=False)
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
        lambda_view_consistency = float(loss_cfg.get("lambda_view_consistency", 0.0))

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
            lambda_view_consistency=lambda_view_consistency,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
