import torch
import torch.nn as nn
import torch.nn.functional as F

# class ClassificationLoss(nn.Module):
#     def __init__(self, penalty_weight=0.5, class_weights=None):
#         super(ClassificationLoss, self).__init__()
#         if class_weights is not None:
#             self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)  # Use class weights
#         else:
#             self.cross_entropy_loss = nn.CrossEntropyLoss()  # No class weights
#         self.penalty_weight = penalty_weight  # Weight of the penalty for incorrect classes

#     def forward(self, logits, true_labels):
#         # Compute weighted cross-entropy loss
#         ce_loss = self.cross_entropy_loss(logits, true_labels)

#         #Compute probabilities from logits
#         probs = torch.softmax(logits, dim=1)  # Shape: (batch_size, num_classes)

#         # Create a mask to ignore the correct class
#         batch_size = logits.shape[0]
#         correct_class_mask = torch.zeros_like(probs)
#         correct_class_mask[torch.arange(batch_size), true_labels] = 1

#         # Penalize the incorrect class probabilities
#         incorrect_probs = probs * (1 - correct_class_mask)  # Mask out correct class probabilities
#         penalty = incorrect_probs.sum(dim=1).mean()  # Mean of the summed incorrect probabilities

#         # Combine weighted cross-entropy loss with the penalty
#         total_loss = ce_loss + self.penalty_weight * penalty

#         return total_loss

#: State-dict prefix of the classification loss inside ``Blip2Qformer``.
CLASSIFICATION_LOSS_STATE_PREFIX = "cls_loss_fn."


def drop_loss_config_state(state_dict, prefix=CLASSIFICATION_LOSS_STATE_PREFIX):
    """``state_dict`` without the classification loss's buffers.

    Checkpoints written before 2026-10-01 carry the CE class-weight vectors as
    ``cls_loss_fn.cross_entropy_loss_list.<i>.weight``. They are configuration,
    not learned state, and now come from the YAML only.
    """
    kept = type(state_dict)(
        (name, value) for name, value in state_dict.items() if not name.startswith(prefix)
    )
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        kept._metadata = metadata
    return kept


def logit_adjustment_offsets(class_counts, num_abnormalities=14, tau=1.0):
    """``[A, 3]`` offsets ``tau * log(prior)`` for the logit-adjusted loss.

    ``class_counts`` holds one ``[n_negative, n_positive, n_uncertain]`` vector
    per abnormality, counted on the TRAIN split under the configured blank
    policy. A zero count gets offset 0 (see :class:`ClassificationLoss`).
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    if counts.shape != (num_abnormalities, 3):
        raise ValueError(
            f"logit adjustment needs [{num_abnormalities}, 3] class counts, "
            f"got {tuple(counts.shape)}"
        )
    if (counts < 0).any() or (counts.sum(dim=1) <= 0).any():
        raise ValueError("class counts must be non-negative with a positive total per row")
    if tau <= 0:
        raise ValueError("logit_adjust_tau must be positive")
    prior = counts / counts.sum(dim=1, keepdim=True)
    offsets = torch.where(
        prior > 0, float(tau) * torch.log(prior.clamp_min(1e-300)), torch.zeros_like(prior)
    )
    return offsets.float()


class ClassificationLoss(nn.Module):
    """Per-abnormality weighted cross entropy over the paper's THREE classes.

    Negative (0), Positive (1) and Uncertain (2) are always three separate
    classes, as the META-CXR paper trains them (Eq. 10). There is deliberately
    no option to fold Uncertain into another class or to drop it: those
    "uncertain_policy" variants turned the task into a binary one and were
    removed on 2026-09-29 (D-023). ``IGNORE_LABEL`` (-100) cells -- studies with
    no CheXpert information -- are the only cells skipped.

    ``penalty_weight`` remains in the signature for old configs.  The previous
    implementation computed that penalty and then discarded it, so it is no
    longer evaluated.

    ``logit_adjust_counts`` (one ``[n_negative, n_positive, n_uncertain]``
    train count vector per abnormality) switches to the logit-adjusted loss of
    Menon et al. (ICLR 2021, Eq. 10): the cross entropy is taken over
    ``logits + tau * log(prior)``, while the model's own logits -- what argmax,
    the exported probabilities and every metric read -- stay unadjusted. It is
    Fisher-consistent for the balanced error, i.e. ``1 - macro_recall``, and
    replaces ``class_weights`` (the two are mutually exclusive). A class with
    zero train count gets offset 0: it is never a target, so cross entropy
    already keeps its raw logit below the present classes'.
    """

    def __init__(
        self,
        penalty_weight=0.0,
        class_weights=None,
        num_abnormalities=14,
        label_smoothing=0.0,
        logit_adjust_counts=None,
        logit_adjust_tau=1.0,
    ):
        super().__init__()
        self.penalty_weight = float(penalty_weight)
        if logit_adjust_counts is not None:
            if class_weights is not None:
                raise ValueError(
                    "logit adjustment replaces class_weights; configure one or the other"
                )
            self.register_buffer(
                "logit_offsets",
                logit_adjustment_offsets(
                    logit_adjust_counts, num_abnormalities, logit_adjust_tau
                ),
                persistent=False,
            )
        else:
            self.logit_offsets = None
        if class_weights is not None:
            if not isinstance(class_weights, (list, tuple)):
                raise TypeError("class_weights must contain one tensor per abnormality")
            if len(class_weights) != num_abnormalities:
                raise ValueError(
                    f"expected {num_abnormalities} class-weight vectors, "
                    f"got {len(class_weights)}"
                )
            weights = [torch.as_tensor(w, dtype=torch.float) for w in class_weights]
        else:
            weights = [None] * num_abnormalities
        self.cross_entropy_loss_list = nn.ModuleList(
            [
                nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
                for w in weights
            ]
        )
        # Loss configuration comes from the YAML, never from a checkpoint
        # (2026-10-01): a checkpoint written with class_weights must load into
        # a logit-adjusted model and vice versa. See drop_loss_config_state.
        for loss_fn, w in zip(self.cross_entropy_loss_list, weights):
            if w is not None:
                loss_fn.register_buffer("weight", w, persistent=False)

    def forward(self, logits, true_labels, sample_mask=None):
        if logits.ndim != 3 or true_labels.ndim != 2:
            raise ValueError("expected logits [B,A,C] and labels [B,A]")
        if logits.shape[:2] != true_labels.shape:
            raise ValueError(
                f"logit/label shape mismatch: {tuple(logits.shape)} vs "
                f"{tuple(true_labels.shape)}"
            )

        if sample_mask is None:
            sample_mask = torch.ones(
                logits.shape[0], dtype=torch.bool, device=logits.device
            )
        else:
            sample_mask = torch.as_tensor(
                sample_mask, device=logits.device, dtype=torch.bool
            ).reshape(-1)
            if sample_mask.numel() != logits.shape[0]:
                raise ValueError("sample_mask must contain one value per batch item")

        losses = []
        for abnormality_idx, loss_fn in enumerate(self.cross_entropy_loss_list):
            labels_i = true_labels[:, abnormality_idx].long()
            # Also accept -100 for future partially-labelled annotations.
            valid = sample_mask & (labels_i >= 0) & (labels_i < logits.shape[-1])
            if valid.any():
                logits_i = logits[valid, abnormality_idx]
                if self.logit_offsets is not None:
                    logits_i = logits_i + self.logit_offsets[abnormality_idx].to(logits_i.dtype)
                losses.append(loss_fn(logits_i, labels_i[valid]))

        if not losses:
            # Keep the zero connected to the graph for backward/DDP.
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


def soft_target_kl_loss(
    student_logits, teacher_logits, sample_mask=None, temperature=2.0
):
    """Distil detached teacher probabilities into image-only student logits."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if sample_mask is None:
        sample_mask = torch.ones(
            student_logits.shape[0], dtype=torch.bool, device=student_logits.device
        )
    else:
        sample_mask = torch.as_tensor(
            sample_mask, dtype=torch.bool, device=student_logits.device
        ).reshape(-1)
    if sample_mask.numel() != student_logits.shape[0]:
        raise ValueError("sample_mask must contain one value per batch item")
    if not sample_mask.any():
        return student_logits.sum() * 0.0

    student_log_prob = F.log_softmax(
        student_logits[sample_mask] / temperature, dim=-1
    )
    teacher_prob = F.softmax(
        teacher_logits[sample_mask].detach() / temperature, dim=-1
    )
    return (
        F.kl_div(student_log_prob, teacher_prob, reduction="none")
        .sum(dim=-1)
        .mean()
        * temperature**2
    )

# class InfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.07, margin=0.5):
#         """
#         Modified InfoNCE Loss to enforce relative positions of Positive, Negative, and Uncertain states.

#         Args:
#             temperature (float): Temperature parameter for scaling similarity logits.
#             margin (float): Margin to enforce separation between Positive and Negative states.
#         """
#         super(InfoNCELoss, self).__init__()
#         self.temperature = temperature
#         self.margin = margin

#     def forward(self, expert_tokens, labels):
#         B, N, D = expert_tokens.shape
#         expert_tokens = F.normalize(expert_tokens, dim=-1)

#         total_loss = 0.0
#         for i in range(N):
#             tokens = expert_tokens[:, i, :]
#             token_labels = labels[:, i]

#             # Masks
#             pos_mask = (token_labels == 1).float()
#             neg_mask = (token_labels == 0).float()
#             # unc_mask = (token_labels == 2).float()

#             pos_indices = pos_mask.nonzero(as_tuple=True)[0]
#             neg_indices = neg_mask.nonzero(as_tuple=True)[0]
#             # unc_indices = unc_mask.nonzero(as_tuple=True)[0]

#             similarity_matrix = torch.matmul(tokens, tokens.T)  # Pairwise similarities

#             # Positive-Negative Separation
#             if len(pos_indices) > 0 and len(neg_indices) > 0:
#                 pos_neg_similarity = similarity_matrix[pos_indices][:, neg_indices]
#                 pos_neg_loss = torch.relu(self.margin - (1 - pos_neg_similarity)).mean()
#             else:
#                 pos_neg_loss = 0.0

#             # # Uncertain Alignment
#             # if len(unc_indices) > 0 and len(pos_indices) > 0 and len(neg_indices) > 0:
#             #     pos_unc_similarity = similarity_matrix[unc_indices][:, pos_indices].mean(dim=1)
#             #     neg_unc_similarity = similarity_matrix[unc_indices][:, neg_indices].mean(dim=1)
#             #     unc_loss = torch.abs(pos_unc_similarity - neg_unc_similarity).mean()
#             # else:
#             #     unc_loss = 0.0

#             # Dynamically weight the contributions
#             # contribution_weight = len(pos_indices) + len(neg_indices) + len(unc_indices) + 1e-6
#             # total_loss += (pos_neg_loss + unc_loss) / contribution_weight

#             total_loss += (pos_neg_loss)
            
#         return total_loss / N


class AttentionPooling(nn.Module):
    def __init__(self, d_embedding, num_abnormalities):
        super().__init__()
        self.query_vectors = nn.Parameter(torch.randn(num_abnormalities, d_embedding))  # Learnable queries
        nn.init.xavier_uniform_(self.query_vectors)

    def forward(self, common_representations):
        """
        Args:
            common_representations: Tensor of shape [batch_size, num_tokens, d_embedding]
        
        Returns:
            pooled_representations: Tensor of shape [batch_size, num_abnormalities, d_embedding]
        """
        batch_size, num_tokens, d_embedding = common_representations.shape
        num_abnormalities = self.query_vectors.size(0)

        # Compute attention scores for each abnormality
        attention_scores = torch.einsum("ad,bnd->ban", self.query_vectors, common_representations)  # [batch_size, num_abnormalities, num_tokens]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_abnormalities, num_tokens]

        # Pool features using attention weights
        pooled_representations = torch.einsum("ban,bnd->bad", attention_weights, common_representations)  # [batch_size, num_abnormalities, d_embedding]

        return pooled_representations

class AbnormalitySpecificLoss(nn.Module):
    def __init__(
        self,
        temperature=0.07,
        margin=0.7,
        d_embedding=768,
        num_abnormalities=14,
    ):
        """
        Modified InfoNCE Loss for abnormality-specific tokens (paper Eqs. 16-19).

        Always three-class: positive/negative separation plus the uncertain
        term that keeps Uncertain samples equidistant from both.

        Args:
            temperature (float): Temperature parameter for scaling similarity logits.
            margin (float): Margin to enforce separation between Positive and Negative states.
            inter_abnormality_weight (float): Weight for inter-abnormality dissimilarity.
        """
        super(AbnormalitySpecificLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.attention_pooling = AttentionPooling(d_embedding, num_abnormalities)
    
    def orthogonality_loss(self, common_representations):
        """
        Compute orthogonality loss for the common tokens.
        
        Args:
            common_representations: Tensor of shape [batch_size, num_tokens, d_embedding]

        Returns:
            orth_loss: Orthogonality loss
        """
        batch_size, num_tokens, d_embedding = common_representations.shape
        common_representations = F.normalize(common_representations, dim=-1)  # Normalize token embeddings

        # Compute pairwise similarity within tokens
        similarity_matrix = torch.einsum("bnd,bmd->bnm", common_representations, common_representations)  # [batch_size, num_tokens, num_tokens]

        # Compute Frobenius norm loss to enforce orthogonality
        off_diagonal_mask = 1 - torch.eye(
            num_tokens,
            device=common_representations.device,
            dtype=common_representations.dtype,
        ).unsqueeze(0)
        # Penalize only off-diagonal elements
        orth_loss = torch.mean((similarity_matrix * off_diagonal_mask) ** 2)
        return orth_loss
    
    def compute_weighted_sparsity_loss(self, attention_weights_list):
        """
        Compute sparsity loss across layers with layer-specific weighting.
        
        Args:
            attention_weights_list (list of torch.Tensor): List of attention weights for each layer.
            lambda_sparsity (float): Global weight for sparsity loss.
        
        Returns:
            sparsity_loss: Weighted sparsity loss across layers.
        """
        total_sparsity_loss = 0.0
        num_layers = len(attention_weights_list)
        
        # Assign higher weights to deeper layers
        layer_weights = torch.sigmoid(
            torch.linspace(
                -2,
                2,
                steps=num_layers,
                device=attention_weights_list[0].device,
                dtype=attention_weights_list[0].dtype,
            )
        )

        for i, layer_attention_weights in enumerate(attention_weights_list):
            # Compute sparsity loss for this layer
            sparsity_loss_layer = -torch.sum(
                layer_attention_weights * torch.log(layer_attention_weights + 1e-6)
            ) / layer_attention_weights.numel()
            
            # Apply layer-specific weight
            total_sparsity_loss += layer_weights[i] * sparsity_loss_layer

        # Scale by lambda_sparsity
        sparsity_loss = total_sparsity_loss / num_layers

        return sparsity_loss


    def forward(
        self,
        common_representations,
        attention_weights_list,
        labels=None,
        sample_mask=None,
    ):
        """
        Args:
            common_representations: Tensor of shape [batch_size, num_tokens, d_embedding]
            labels: Tensor of shape [batch_size, num_abnormalities] (binary labels per abnormality)

        Returns:
            total_loss: Combined loss across all abnormalities
        """
        pooled_representations_ = self.attention_pooling(common_representations)
        orth_loss = self.orthogonality_loss(common_representations)
        sparsity_loss = self.compute_weighted_sparsity_loss(attention_weights_list)
        
        zero = common_representations.sum() * 0.0
        if labels is None:
            return pooled_representations_, orth_loss, zero, sparsity_loss

        if sample_mask is not None:
            sample_mask = torch.as_tensor(
                sample_mask, dtype=torch.bool, device=labels.device
            ).reshape(-1)
            if sample_mask.numel() != labels.shape[0]:
                raise ValueError("sample_mask must contain one value per batch item")
            pooled_for_loss = pooled_representations_[sample_mask]
            labels_for_loss = labels[sample_mask]
        else:
            pooled_for_loss = pooled_representations_
            labels_for_loss = labels

        if pooled_for_loss.shape[0] == 0:
            return pooled_representations_, orth_loss, zero, sparsity_loss

        batch_size, num_abnormalities, d_embedding = pooled_for_loss.shape
        # AMP can make very small fp16 norms underflow; normalizing in fp32
        # keeps cosine similarities in [-1, 1] and this loss mathematically
        # bounded instead of allowing it to dominate every other objective.
        pooled_representations = F.normalize(
            pooled_for_loss.float(), dim=-1
        )

        contrastive_loss = zero

        # Loop over each abnormality
        for a in range(num_abnormalities):
            tokens = pooled_representations[:, a, :]  # [batch_size, d_embedding]
            token_labels = labels_for_loss[:, a]  # [batch_size]

            # Masks
            pos_mask = (token_labels == 1).float()
            neg_mask = (token_labels == 0).float()
            unc_mask = (token_labels == 2).float()
            
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            neg_indices = neg_mask.nonzero(as_tuple=True)[0]
            unc_indices = unc_mask.nonzero(as_tuple=True)[0]

            # Compute pairwise similarity matrix
            similarity_matrix = torch.matmul(tokens, tokens.T)  # [batch_size, batch_size]

            # Positive-Negative Separation
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_neg_similarity = similarity_matrix[pos_indices][:, neg_indices]  # [num_pos, num_neg]
                pos_neg_loss = torch.relu(self.margin - (1 - pos_neg_similarity)).mean()
            else:
                pos_neg_loss = zero
            
            # Uncertain Alignment
            if (
                len(unc_indices) > 0
                and len(pos_indices) > 0
                and len(neg_indices) > 0
            ):
                pos_unc_similarity = similarity_matrix[unc_indices][:, pos_indices].mean(dim=1)
                neg_unc_similarity = similarity_matrix[unc_indices][:, neg_indices].mean(dim=1)
                unc_loss = torch.abs(pos_unc_similarity - neg_unc_similarity).mean()
            else:
                unc_loss = zero

            # Do not mutate ``zero`` in place. Missing-class branches reuse that
            # tensor; ``+=`` made them reuse the accumulated loss and doubled
            # it repeatedly across later pathologies.
            contrastive_loss = contrastive_loss + pos_neg_loss + unc_loss

        contrastive_loss = contrastive_loss / num_abnormalities
        
        return pooled_representations_, orth_loss, contrastive_loss, sparsity_loss



class AttentionLoss:
    """
    Computes combined attention consistency loss and sparsity loss.
    This class allows modular computation of these losses for attention weights.

    Args:
        lambda_sparsity (float): Weighting factor for the sparsity loss component.
    """
    def __init__(self, lambda_sparsity=0.3):
        self.lambda_sparsity = lambda_sparsity

    def compute_consistency_loss(self, attention_weights_list):
        """
        Computes the attention consistency loss for each expert token across the batch.

        Args:
            attention_weights_list: List of attention weights [batch_size, num_expert_tokens, num_image_patches].

        Returns:
            consistency_loss: Scalar loss enforcing consistent attention for each expert token.
        """
        consistency_loss = 0.0
        num_layers = len(attention_weights_list)

        for attention_weights in attention_weights_list:  # Iterate over layers
            num_tokens = attention_weights.size(1)  # Number of expert tokens

            for token_idx in range(num_tokens):  # Iterate over expert tokens
                # Extract attention weights for this token: [batch_size, num_image_patches]
                token_attention = attention_weights[:, token_idx, :]

                # Consistency Loss: Mean Squared Deviation from Batch Mean
                mean_attention = token_attention.mean(dim=0, keepdim=True)  # [1, num_image_patches]
                deviation = token_attention - mean_attention
                consistency_loss += torch.mean(deviation ** 2)  # MSE loss for this token

        # Normalize by the number of layers and tokens
        return consistency_loss / (num_layers * attention_weights_list[0].size(1))

    def compute_sparsity_loss(self, attention_weights_list):
        """
        Computes sparsity loss to encourage focused attention maps.

        Args:
            attention_weights_list: List of attention weights [batch_size, num_expert_tokens, num_image_patches].

        Returns:
            sparsity_loss: Scalar loss encouraging sparse attention maps.
        """
        sparsity_loss = 0.0
        num_layers = len(attention_weights_list)

        for attention_weights in attention_weights_list:  # Iterate over layers
            num_tokens = attention_weights.size(1)  # Number of expert tokens

            for token_idx in range(num_tokens):  # Iterate over expert tokens
                # Extract attention weights for this token: [batch_size, num_image_patches]
                token_attention = attention_weights[:, token_idx, :]

                # Sparsity Loss: L1 Regularization of Attention Weights
                sparsity_loss += torch.sum(torch.abs(token_attention)) / token_attention.size(0)  # Average over batch

        # Normalize by the number of layers and tokens
        return sparsity_loss / (num_layers * attention_weights_list[0].size(1))

    def __call__(self, attention_weights_list):
        """
        Computes the total loss as the sum of consistency loss and sparsity loss.

        Args:
            attention_weights_list: List of attention weights [batch_size, num_expert_tokens, num_image_patches].

        Returns:
            total_loss: Combined loss (attention consistency + sparsity).
            consistency_loss: Attention consistency loss component.
            sparsity_loss: Sparsity loss component.
        """
        consistency_loss = self.compute_consistency_loss(attention_weights_list)
        sparsity_loss = self.compute_sparsity_loss(attention_weights_list)
        total_loss = consistency_loss + self.lambda_sparsity * sparsity_loss

        return total_loss


class MultiPositiveContrastiveLoss(nn.Module):
    """Multi-positive InfoNCE over the pre-fusion visual representations.

    Every image of a study (anchor + its auxiliary views) is pooled to one
    vector. For each anchor, the auxiliary views of the *same* study are
    positives and every image of the other studies in the batch is a negative.
    The number of positives varies per study, hence the multi-positive form:

        L_i = -1/|P(i)| * sum_{p in P(i)} log( exp(s_ip/T) / sum_{a != i} exp(s_ia/T) )

    Anchors with no auxiliary view contribute nothing.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, aux, aux_mask):
        """
        anchor:   [B, D]      pooled + projected anchor vector
        aux:      [B, N, D]   pooled + projected auxiliary vectors (padded)
        aux_mask: [B, N] bool True = real view

        ⚠ Signature changed 2026-08-16. It used to take token sequences
        (``[B,P,D]`` / ``[B,N,P,D]``) and mean-pool them itself. Pooling now
        happens in the caller because the two encoders need different pooling —
        PubMedCLIP has a real CLS token, BioViL does not — and because the
        vectors must pass a projection head before being contrasted. Mean-pooling
        the raw frozen features here is exactly what made this loss a constant.
        """
        if aux is None or aux.shape[1] == 0 or aux_mask is None:
            return anchor.new_zeros(())
        if not aux_mask.any():
            return anchor.new_zeros(())

        B, N = aux_mask.shape
        device = anchor.device
        aux_mask = aux_mask.to(device=device, dtype=torch.bool)

        if anchor.ndim != 2 or aux.ndim != 3:
            raise ValueError(
                "expected pooled vectors: anchor [B,D] and aux [B,N,D]; got "
                f"{tuple(anchor.shape)} and {tuple(aux.shape)}"
            )
        a_vec = F.normalize(anchor, dim=-1)                          # [B, D]
        x_vec = F.normalize(aux, dim=-1).reshape(B * N, -1)          # [B*N, D]

        # Candidate pool: every anchor, then every auxiliary slot.
        cand = torch.cat([a_vec, x_vec], dim=0)                     # [M, D]
        cand_study = torch.cat([
            torch.arange(B, device=device),
            torch.arange(B, device=device).repeat_interleave(N),
        ])
        cand_valid = torch.cat([
            torch.ones(B, dtype=torch.bool, device=device),
            aux_mask.reshape(B * N),
        ])

        sim = (a_vec @ cand.t()) / self.temperature                 # [B, M]
        rows = torch.arange(B, device=device)
        is_self = torch.zeros_like(cand_valid).repeat(B, 1)
        is_self[rows, rows] = True                                  # anchor vs itself

        usable = cand_valid.unsqueeze(0) & ~is_self
        positives = usable & (cand_study.unsqueeze(0) == rows.unsqueeze(1))

        # log-softmax over the usable candidates only.
        sim = sim.masked_fill(~usable, float("-inf"))
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = log_prob.masked_fill(~positives, 0.0)

        n_pos = positives.sum(dim=1)
        has_pos = n_pos > 0
        if not has_pos.any():
            return anchor.new_zeros(())
        per_anchor = -log_prob.sum(dim=1)[has_pos] / n_pos[has_pos]
        return per_anchor.mean()


def view_consistency_loss(
    fused_logits,
    anchor_logits,
    has_aux,
    margin=0.0,
    confidence_gate=False,
    gate_tolerance=0.0,
):
    """Soft, conditional agreement between the fused and anchor-only predictions.

    The original form was an unconditional symmetric KL, justified as "adding
    views must not change *which* abnormalities are predicted". That premise is
    wrong for this dataset: a lateral view exists precisely to show what the
    frontal cannot, and 55% of studies have one. Forcing the fused prediction
    onto the anchor-only prediction penalises the model for *using* the extra
    view, which is the opposite of what multi-view fusion is for.

    Two knobs relax it, and both default to off so the historical behaviour is
    reproducible for ablation:

    ``margin``
        Divergence below this costs nothing (hinge). Small drift is normal
        re-weighting, not contradiction; only real flips should be charged.

    ``confidence_gate``
        Waive the penalty on cells where the fused distribution is *more*
        confident (lower entropy) than the anchor-only one by more than
        ``gate_tolerance`` nats. Sharpening is the signature of new evidence;
        smearing is the signature of noise. The gate is **detached** — it
        selects where the loss applies and must not itself carry gradient, or
        the model could minimise the term by manipulating the gate instead of
        the prediction.

    With ``margin=0.0`` and ``confidence_gate=False`` this returns exactly the
    previous value.

    fused_logits / anchor_logits: [B, num_abnormalities, num_classes]
    has_aux: [B] bool
    """
    if has_aux is None or not has_aux.any():
        return fused_logits.new_zeros(())
    margin = float(margin)
    if margin < 0.0:
        raise ValueError("margin must be >= 0")
    gate_tolerance = float(gate_tolerance)
    if gate_tolerance < 0.0:
        raise ValueError("gate_tolerance must be >= 0")

    p = F.log_softmax(fused_logits[has_aux], dim=-1)
    q = F.log_softmax(anchor_logits[has_aux], dim=-1)
    kl_pq = F.kl_div(q, p, log_target=True, reduction="none").sum(-1)
    kl_qp = F.kl_div(p, q, log_target=True, reduction="none").sum(-1)
    divergence = 0.5 * (kl_pq + kl_qp)

    if margin > 0.0:
        divergence = F.relu(divergence - margin)

    if not confidence_gate:
        return divergence.mean()

    # Entropy per (study, abnormality); lower means more confident.
    entropy_fused = -(p.exp() * p).sum(-1)
    entropy_anchor = -(q.exp() * q).sum(-1)
    # Charge the cell unless fusing made it decisively more confident.
    keep = (entropy_fused >= entropy_anchor - gate_tolerance).detach().to(
        divergence.dtype
    )
    kept = keep.sum()
    if kept.item() == 0:
        return fused_logits.new_zeros(())
    return (divergence * keep).sum() / kept






"""
# Example usage
logits = torch.tensor([[2.0, 1.0, 0.1], [0.1, 2.0, 0.9]])  # Logits from the model
true_labels = torch.tensor([0, 1])  # True labels (class 0 and 1)

# Define class weights (e.g., based on class imbalance in the dataset)
# In this case, we assign higher weight to class 2 (assumed to be the rare class)
class_weights = torch.tensor([1.0, 1.0, 2.0])  # Class 2 has double the weight

# Initialize and compute custom loss with class weights
loss_fn = ClassificationLoss(penalty_weight=0.5, class_weights=class_weights)
loss = loss_fn(logits, true_labels)

print(f"Loss with class weighting: {loss.item()}")
"""


def smoothed_cross_entropy(logits, targets, label_smoothing):
    """Cross entropy whose smoothing mass goes to the finite logits only.

    ``F.cross_entropy(label_smoothing=eps)`` spreads eps over EVERY column, and
    ITC masks invalid candidates (studies without usable FINDINGS) with -inf, so
    the built-in turns the loss into inf. With every column finite this equals
    ``F.cross_entropy(logits, targets, label_smoothing=eps)``.
    """
    if label_smoothing <= 0:
        return F.cross_entropy(logits, targets)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    finite = torch.isfinite(logits)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    safe = log_probs.masked_fill(~finite, 0.0)
    smooth = -safe.sum(dim=1) / finite.sum(dim=1).clamp_min(1)
    return ((1.0 - label_smoothing) * nll + label_smoothing * smooth).mean()


def siglip_loss(sim, logit_scale, bias, valid=None):
    """SigLIP pairwise sigmoid loss (Zhai et al., arXiv 2303.15343).

    ``sim`` [N, N] holds raw image-text similarities (here the max over the
    Q-Former query tokens), true pairs on the diagonal. Every pair is scored on
    its own -- no softmax normalisation over the batch -- which is what makes
    small batches degrade far less than InfoNCE:

        L = -1/n * sum_ij log sigmoid(z_ij * (exp(logit_scale) * s_ij + bias)),
        z_ij = +1 on the diagonal, -1 elsewhere,

    over the ``valid`` rows AND columns only (studies without usable FINDINGS
    are neither queries nor candidates).
    """
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"siglip_loss needs a square [N, N] similarity, got {tuple(sim.shape)}")
    if valid is not None:
        keep = torch.as_tensor(valid, dtype=torch.bool, device=sim.device).reshape(-1)
        sim = sim[keep][:, keep]
    n = sim.shape[0]
    if n == 0:
        return sim.sum() * 0.0
    logits = sim.float() * logit_scale.float().exp() + bias.float()
    signs = 2.0 * torch.eye(n, device=sim.device, dtype=logits.dtype) - 1.0
    return -F.logsigmoid(signs * logits).sum() / n
