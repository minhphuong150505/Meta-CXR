"""Reference outputs of PubMedCLIP's own CLIPImageProcessor (D-022).

Runs the slow ``CLIPImageProcessor`` loaded from
``flaviagiammarino/pubmed-clip-vit-base-patch32`` on SYNTHETIC grayscale
radiograph-like images (no patient data) and stores the pixel values, so
tests/test_pubmedclip_preprocess.py can pin
``vision_encoders/pubmedclip/preprocess.pubmedclip_preprocess`` against it on a
box without transformers. Run on the training host:

    python scripts/make_pubmedclip_preprocess_reference.py \
        tests/fixtures/pubmedclip_preprocess_reference.npz
"""
import sys

import numpy as np
import transformers
from PIL import Image
from transformers import CLIPImageProcessor

MODEL = "flaviagiammarino/pubmed-clip-vit-base-patch32"


def synthetic(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(xx / 17.0) + np.cos(yy / 23.0)) * 60 + 120
    blob = 80 * np.exp(-(((xx - w * 0.6) ** 2 + (yy - h * 0.4) ** 2) / (2 * (min(h, w) / 6) ** 2)))
    noise = rng.normal(0, 6, size=(h, w))
    return Image.fromarray(np.clip(base + blob + noise, 0, 255).astype(np.uint8), mode="L")


# Portrait, landscape, square, smaller than 224, and a 2544x3056-like aspect.
SHAPES = [(300, 200, 0), (180, 320, 1), (224, 224, 2), (150, 100, 3), (611, 509, 4)]

if __name__ == "__main__":
    processor = CLIPImageProcessor.from_pretrained(MODEL)
    out = {"transformers_version": np.array(transformers.__version__)}
    for i, (h, w, seed) in enumerate(SHAPES):
        px = processor(images=synthetic(h, w, seed).convert("RGB"), return_tensors="np")
        out[f"out_{i}"] = px["pixel_values"][0].astype(np.float32)
        out[f"shape_{i}"] = np.array([h, w, seed])
    np.savez_compressed(sys.argv[1], **out)
    print("wrote", sys.argv[1], transformers.__version__,
          [out[f"out_{i}"].shape for i in range(len(SHAPES))])
