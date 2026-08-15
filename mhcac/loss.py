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

class ClassificationLoss(nn.Module):
    """Per-abnormality weighted cross entropy with sample-level masking.

    ``penalty_weight`` remains in the signature for old configs.  The previous
    implementation computed that penalty and then discarded it, so it is no
    longer evaluated.
    """

    def __init__(
        self,
        penalty_weight=0.0,
        class_weights=None,
        num_abnormalities=14,
        label_smoothing=0.0,
        uncertain_policy="three_class",
    ):
        super().__init__()
        valid_policies = {
            "three_class",
            "uncertain_as_positive",
            "uncertain_as_negative",
            "ignore_uncertain",
        }
        if uncertain_policy not in valid_policies:
            raise ValueError(
                f"unknown uncertain_policy {uncertain_policy!r}; expected one of "
                f"{', '.join(sorted(valid_policies))}"
            )
        self.uncertain_policy = uncertain_policy
        self.penalty_weight = float(penalty_weight)
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
            if self.uncertain_policy == "ignore_uncertain":
                valid = valid & (labels_i != 2)
            elif self.uncertain_policy == "uncertain_as_positive":
                labels_i = torch.where(labels_i == 2, 1, labels_i)
            elif self.uncertain_policy == "uncertain_as_negative":
                labels_i = torch.where(labels_i == 2, 0, labels_i)
            if valid.any():
                losses.append(loss_fn(logits[valid, abnormality_idx], labels_i[valid]))

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
        uncertain_policy="three_class",
    ):
        """
        Modified InfoNCE Loss for abnormality-specific tokens.

        Args:
            temperature (float): Temperature parameter for scaling similarity logits.
            margin (float): Margin to enforce separation between Positive and Negative states.
            inter_abnormality_weight (float): Weight for inter-abnormality dissimilarity.
        """
        super(AbnormalitySpecificLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        valid_policies = {
            "three_class",
            "uncertain_as_positive",
            "uncertain_as_negative",
            "ignore_uncertain",
        }
        if uncertain_policy not in valid_policies:
            raise ValueError(f"unknown uncertain_policy {uncertain_policy!r}")
        self.uncertain_policy = uncertain_policy
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

            if self.uncertain_policy == "uncertain_as_positive":
                token_labels = torch.where(token_labels == 2, 1, token_labels)
            elif self.uncertain_policy == "uncertain_as_negative":
                token_labels = torch.where(token_labels == 2, 0, token_labels)

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
                self.uncertain_policy != "ignore_uncertain"
                and len(unc_indices) > 0
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


class MentionGateLoss(nn.Module):
    """"Will the report mention this finding at all?", one binary head per label.

    This is the only consumer of the 79.5% of the CheXpert matrix that is blank.
    Everything else masks those cells out, which is correct for a Positive /
    Negative / Uncertain question -- a blank is not a negative -- but it leaves
    the model unable to say "I have nothing to report here". Forced to pick one
    of three classes for all fourteen findings on every image, it emitted an
    average of 10.8 positives per study on the test split, and called every
    single study `No Finding = Positive` while simultaneously flagging 8.8 other
    findings on it.

    ``pos_weights`` carries (n_not_mentioned / n_mentioned) x kappa per label, so
    a wrongly silent gate -- the model hiding a finding the radiologist wrote
    about -- costs kappa times a gate that speaks up unnecessarily. The cap
    matters as much as the weights: three labels are mentioned so rarely that
    their raw ratio is 26-79, and applying that unclipped just moves the
    degenerate "always the majority class" behaviour into the gate.

    ``sample_mask`` drops studies that matched no CheXpert record. Their blank
    pattern is unknown, not empty, and training them as fourteen zeros would
    teach the gate to stay silent on exactly the rows with no supervision.
    """

    def __init__(self, num_abnormalities=14, pos_weights=None, weight_cap=10.0):
        super().__init__()
        self.num_abnormalities = num_abnormalities
        if pos_weights is None:
            weights = torch.ones(num_abnormalities)
        else:
            if len(pos_weights) != num_abnormalities:
                raise ValueError(
                    f"pos_weights must hold one value per abnormality: "
                    f"got {len(pos_weights)} for {num_abnormalities}"
                )
            weights = torch.tensor([float(w) for w in pos_weights])
            if (weights <= 0).any():
                raise ValueError("pos_weights must be positive")
            weights = weights.clamp(max=float(weight_cap))
        self.register_buffer("pos_weight", weights)

    def forward(self, logits, targets, sample_mask=None):
        if logits.shape != targets.shape:
            raise ValueError(
                f"gate logit/target shape mismatch: {tuple(logits.shape)} vs "
                f"{tuple(targets.shape)}"
            )
        if logits.shape[1] != self.num_abnormalities:
            raise ValueError(
                f"expected {self.num_abnormalities} abnormalities, got {logits.shape[1]}"
            )

        if sample_mask is None:
            sample_mask = torch.ones(
                logits.shape[0], dtype=torch.bool, device=logits.device
            )
        else:
            sample_mask = torch.as_tensor(
                sample_mask, dtype=torch.bool, device=logits.device
            ).reshape(-1)
            if sample_mask.numel() != logits.shape[0]:
                raise ValueError("sample_mask must contain one value per batch item")
        if not sample_mask.any():
            # Keep the zero connected to the graph for backward/DDP.
            return logits.sum() * 0.0

        per_cell = F.binary_cross_entropy_with_logits(
            logits[sample_mask],
            targets[sample_mask].to(logits.dtype),
            pos_weight=self.pos_weight.to(logits.dtype),
            reduction="none",
        )
        return per_cell.mean()


# Class index convention, shared with ClassificationLoss and the evaluator:
#   0 = Negative, 1 = Positive, 2 = Uncertain
_NEGATIVE, _POSITIVE, _UNCERTAIN = 0, 1, 2


def mention_marginal_log_probs(conditional_logits, mention_logits):
    """Marginalise the mention gate into the classification distribution.

    The gate and the classifier used to be two independent heads: the gate could
    say "this finding is never mentioned" while the classifier said "Positive",
    and nothing reconciled them, because the gate's prediction was consumed by
    its own BCE and by nothing else. Measured consequence on the 2026-08-15 run:
    macro specificity 0.2637, with specificity ~0 on four labels.

    Making the two hierarchical is what actually couples them::

        P(Negative)  = (1 - m) + m * q_negative
        P(Positive)  =           m * q_positive
        P(Uncertain) =           m * q_uncertain

    where ``m = sigmoid(mention_logits)`` and ``q = softmax(conditional_logits)``.
    "Not mentioned" maps onto Negative because that is what an absent finding
    means in a report. Silence now suppresses positives instead of sitting
    beside them.

    Computed in log space throughout: ``logsigmoid`` and ``log_softmax`` are
    stable where ``log(sigmoid(x))`` is not, and the returned tensor is meant to
    be used exactly like the old logits — ``softmax`` of it recovers the
    marginals, since they already sum to one.

    conditional_logits: [B, A, C>=2]   mention_logits: [B, A]
    returns:            [B, A, C]      log of the marginal probabilities
    """
    if conditional_logits.ndim != 3:
        raise ValueError("conditional_logits must be [B, A, C]")
    if mention_logits.shape != conditional_logits.shape[:2]:
        raise ValueError(
            f"mention_logits must be [B, A]; got {tuple(mention_logits.shape)} "
            f"against {tuple(conditional_logits.shape[:2])}"
        )
    num_classes = conditional_logits.shape[-1]
    if num_classes < 2:
        raise ValueError("need at least Negative and Positive classes")

    log_m = F.logsigmoid(mention_logits).unsqueeze(-1)          # [B,A,1]
    log_not_m = F.logsigmoid(-mention_logits).unsqueeze(-1)     # [B,A,1]
    log_q = F.log_softmax(conditional_logits.float(), dim=-1)   # [B,A,C]

    log_joint = log_m + log_q                                   # mentioned path
    negative = torch.logaddexp(
        log_not_m.squeeze(-1), log_joint[..., _NEGATIVE]
    )
    parts = [
        negative if index == _NEGATIVE else log_joint[..., index]
        for index in range(num_classes)
    ]
    return torch.stack(parts, dim=-1)


class MentionConditionedClassificationLoss(nn.Module):
    """One hierarchical likelihood in place of a gate BCE plus a weighted CE.

    Replaces ``ClassificationLoss`` + ``MentionGateLoss``, which optimised two
    heads that never met::

        not mentioned      ->  -log(1 - m)
        mentioned, class y ->  -log(m) - log(q[y])

    **No inverse-frequency or clinical-kappa weights.** Those weights were an
    attempt to buy a decision preference inside the likelihood, and the run they
    produced shows what that costs: recall 0.9021 against precision 0.6835, four
    labels at specificity ~0. An operating point belongs in the calibrated
    thresholds, which this project already fits on validation after training;
    the training objective should estimate probabilities, not pick a threshold.

    An ignored class cell (uncertain policy, or a blank the labeler never wrote)
    still trains the **mention** term — whether a finding was written about is
    known even when its polarity is not. Only the conditional class term drops.
    """

    def forward(
        self,
        conditional_logits,
        mention_logits,
        labels,
        mention_targets,
        sample_mask=None,
    ):
        if conditional_logits.ndim != 3:
            raise ValueError("conditional_logits must be [B, A, C]")
        if mention_logits.shape != conditional_logits.shape[:2]:
            raise ValueError("mention_logits must be [B, A]")
        if labels.shape != conditional_logits.shape[:2]:
            raise ValueError("labels must be [B, A]")
        if mention_targets.shape != conditional_logits.shape[:2]:
            raise ValueError("mention_targets must be [B, A]")

        device = conditional_logits.device
        labels = labels.to(device)
        mention_targets = mention_targets.to(device=device)
        if sample_mask is None:
            rows = torch.ones(
                conditional_logits.shape[0], dtype=torch.bool, device=device
            )
        else:
            rows = torch.as_tensor(
                sample_mask, dtype=torch.bool, device=device
            ).reshape(-1)
            if rows.numel() != conditional_logits.shape[0]:
                raise ValueError("sample_mask must hold one value per batch item")

        zero = conditional_logits.sum() * 0.0
        if not rows.any():
            return zero

        log_m = F.logsigmoid(mention_logits)
        log_not_m = F.logsigmoid(-mention_logits)
        mentioned = mention_targets > 0.5
        active = rows[:, None].expand_as(mentioned)

        # Mention term: every cell of every supervised study.
        mention_term = torch.where(mentioned, -log_m, -log_not_m)
        mention_term = mention_term * active.to(mention_term.dtype)
        mention_count = active.sum()

        # Conditional term: only mentioned cells whose class survived masking.
        log_q = F.log_softmax(conditional_logits.float(), dim=-1)
        usable = active & mentioned & (labels >= 0) & (labels < log_q.shape[-1])
        if usable.any():
            picked = log_q.gather(
                -1, labels.clamp_min(0).unsqueeze(-1).long()
            ).squeeze(-1)
            class_term = (-picked * usable.to(picked.dtype)).sum() / usable.sum()
        else:
            class_term = zero

        return mention_term.sum() / mention_count.clamp_min(1) + class_term
