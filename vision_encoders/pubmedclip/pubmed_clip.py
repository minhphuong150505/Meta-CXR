import torch
from torch import nn
from transformers import CLIPImageProcessorFast, CLIPModel

class Pubmedclip(nn.Module):
    def __init__(self, aug = None, device=None, project=True):
        """``project=False`` skips the 768->1408 MLP head.

        SharedVisualTokenProjector now owns every encoder's projection to the
        shared visual dimension, so callers that merge downstream must not build
        this head: it would be a trainable module that receives no gradient and
        costs a projection on every forward.
        """
        super(Pubmedclip, self).__init__()  # Initialize nn.Module
        # RESOLVE the device, do not hardcode it. `device='cuda'` made this the
        # one module in the vision stack that could not be built on a CPU box,
        # and the casualty was `scripts/check_itc_gate.py --device cpu` -- the
        # script whose entire job is to decide, cheaply, whether the
        # vision-language objectives are worth a 33 h run. It died in this
        # constructor with "No CUDA GPUs are available" before reading a single
        # image. It also broke the repo rule that nothing hardcodes cuda
        # (see runtime/device.py).
        #
        # Both uses of self.device are init-time `.to()` calls and forward()
        # never reads it, so on a machine with a GPU this resolves to 'cuda' and
        # the training path is byte-for-byte what it was.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load the pre-trained PubMedCLIP model and processor
        self.device = device
        self.aug = aug
        self.project = project
        self.model_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        # use_safetensors is required, not a preference. transformers >= 4.53
        # refuses torch.load on torch < 2.6 (CVE-2025-32434), and the cached
        # main ref for this repo resolves to a snapshot that ships only
        # pytorch_model.bin. Without this the whole run dies in __init__ before
        # a single batch, which is how it was found. Verified on the training
        # host: loads 151.3M parameters with the flag, ValueError without it.
        self.model = CLIPModel.from_pretrained(
            self.model_name, use_safetensors=True
        ).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        # use_fast keeps preprocessing on the GPU. The slow processor converts
        # every batch to CPU numpy, resizes there, and hands back a CPU tensor
        # that forward() then copies to the device again. Measured on the host
        # at batch 6, 448x448: 55.5 ms/batch against 0.3 ms, and the step is
        # ~570 ms -- roughly a tenth of training time spent resizing images on
        # the CPU, about six hours of a ten-epoch run.
        #
        # Outputs differ slightly (max 0.083, mean 0.0076 in normalised units)
        # because the resampling implementations differ. That is a preprocessing
        # change, so features shift a little; it must be the same at train and
        # inference time, which it is, both going through this class.
        # Only the image side is used -- forward() reads pixel_values and
        # never touches the tokenizer -- so take the image processor directly.
        # Going through CLIPProcessor(use_fast=True) instead raises
        # AttributeError: 'CLIPImageProcessorFast' has no '_valid_processor_keys'
        # on transformers 4.53; the wrapper still assumes the slow class.
        self.processor = CLIPImageProcessorFast.from_pretrained(self.model_name)
        
        # Define the MLP to project patch embeddings to 1408 dimensions
        self.mlp = (
            nn.Sequential(
                nn.Linear(768, 1024),  # Bottleneck layer: from 768 (input) to 1024 (hidden)
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1408)  # Final projection to 1408 dimensions
            ).to(self.device)
            if project
            else None
        )

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, image, apply_aug = True):
        # Input is already a [0,1] float tensor from dataset ToTensor(); skip
        # the processor's /255 rescale or it produces near-constant features.
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to device
        
        # inputs = inputs['pixel_values'].squeeze(0)
        inputs = inputs['pixel_values']

        if apply_aug and self.aug is not None:
             inputs = self.aug(inputs)
             
        # Obtain patch embeddings.
        #
        # Two corrections happen here, both measured on real studies from this
        # dataset (16-32 images, CPU, 2026-08-14). Without them PubMedCLIP
        # contributes almost nothing to MHCAC.
        #
        # 1. post_layernorm. HF returns ``last_hidden_state`` for the vision
        #    tower WITHOUT it (modeling_clip.py:763-765 applies it only to the
        #    pooled CLS). CLIP's ViT is pre-LN, so that tensor is the raw
        #    residual stream -- the one representation the model was never meant
        #    to expose. token 0 is also the vector CLIP was contrastively
        #    trained on, and it is only meaningful after this LayerNorm.
        #
        # 2. Removing the DC component from the patch tokens. The 49 patch
        #    tokens shared a mean pairwise cosine of 0.674, i.e. two thirds of
        #    every token was one fixed direction carrying no spatial
        #    information: attention over them came out nearly flat, so the whole
        #    stream acted as a constant bias. 97% of that direction is constant
        #    across images (cos 0.970 between the per-image mean and the dataset
        #    mean), so it is an offset, not content. Subtracting the per-image
        #    mean takes the cosine to -0.016; post_layernorm alone only reaches
        #    0.587 and does not fix it.
        #
        # The split is deliberate and is what makes the two encoders
        # complementary: token 0 carries the global view, and the patches now
        # carry purely local deviation from it.
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs)
            hidden = self.model.vision_model.post_layernorm(
                vision_outputs.last_hidden_state
            )
            cls_token, patches = hidden[:, :1, :], hidden[:, 1:, :]
            patches = patches - patches.mean(dim=1, keepdim=True)
            patch_embeddings = torch.cat([cls_token, patches], dim=1)

        if self.mlp is None:
            return patch_embeddings, None

        # Project the patch embeddings to 1408 dimensions using the MLP
        batch_size, num_patches, embedding_dim = patch_embeddings.shape  # Expected: (batch_size, num_patches, 768)
        patch_embeddings_clone = patch_embeddings.view(batch_size * num_patches, embedding_dim)  # Flatten for MLP
        projected_embeddings = self.mlp(patch_embeddings_clone)  # Project to 1408 dimensions
        projected_embeddings = projected_embeddings.view(batch_size, num_patches, -1)  # Reshape back to (batch_size, num_patches, 1408)

        return patch_embeddings, projected_embeddings
