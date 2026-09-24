"""Reference outputs of MedCLIPFeatureExtractor under transformers 4.24.0.

The transformation steps of MedCLIPFeatureExtractor (__init__, __call__ and
pad_img) are copied from medclip 0.0.3 (medclip/dataset.py), with only the
input-type validation omitted, so torch/torchvision/nltk are not needed: the
work is done by transformers 4.24's own image mixins, which is the point.
Inputs are synthetic images generated from a seed -- no patient data.

Run in a venv with transformers==4.24.0 (medclip's pin), numpy<2, pillow:

    python scripts/make_medclip_preprocess_reference.py \
        tests/fixtures/medclip_preprocess_reference.npz

tests/test_medclip_swin.py re-generates the same images and compares.
"""
import sys

import numpy as np
import transformers
from PIL import Image
from transformers import CLIPFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature

IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304

class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, do_resize=True, size=224, resample=Image.BICUBIC, do_center_crop=True,
                 crop_size=224, do_normalize=True, image_mean=IMG_MEAN, image_std=IMG_STD,
                 do_convert_rgb=False, do_pad_square=True, **kwargs):
        super().__init__(do_resize, size, resample, do_center_crop, crop_size, do_normalize, image_mean, image_std, do_convert_rgb, **kwargs)
        self.do_pad_square = do_pad_square
    def __call__(self, images, return_tensors=None, **kwargs):
        is_batched = isinstance(images, (list, tuple))
        if not is_batched:
            images = [images]
        if self.do_convert_rgb:
            images = [self.convert_rgb(image) for image in images]
        if self.do_pad_square:
            images = [self.pad_img(image, min_size=self.size) for image in images]
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        return BatchFeature(data={"pixel_values": images_}, tensor_type=return_tensors)
    def pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

def synthetic(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(xx / 17.0) + np.cos(yy / 23.0)) * 60 + 120
    blob = 80 * np.exp(-(((xx - w * 0.6) ** 2 + (yy - h * 0.4) ** 2) / (2 * (min(h, w) / 6) ** 2)))
    noise = rng.normal(0, 6, size=(h, w))
    return Image.fromarray(np.clip(base + blob + noise, 0, 255).astype(np.uint8), mode="L")

SHAPES = [(300, 200, 0), (180, 320, 1), (224, 224, 2), (150, 100, 3)]
fe = MedCLIPFeatureExtractor()
out = {"transformers_version": np.array(transformers.__version__)}
for i, (h, w, seed) in enumerate(SHAPES):
    px = fe(images=synthetic(h, w, seed), return_tensors="np")["pixel_values"][0]
    out[f"out_{i}"] = px.astype(np.float32)
    out[f"shape_{i}"] = np.array([h, w, seed])
np.savez_compressed(sys.argv[1], **out)
print("wrote", sys.argv[1], transformers.__version__, [out[f"out_{i}"].shape for i in range(len(SHAPES))])
