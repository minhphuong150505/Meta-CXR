import dataclasses
import json
import os
import re
from time import time
from enum import Enum, auto
from pathlib import Path
from typing import List, Any

from local_config import (
    JAVA_HOME, JAVA_PATH,
    SPLIT_CSV, REPORTS_CSV, CHEXPERT_CSV, METADATA_CSV,
    PROCESSED_TRAIN_CSV, PROCESSED_VAL_CSV, PROCESSED_TEST_CSV,
)

# set java path
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = JAVA_PATH + os.environ["PATH"]
os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), "gradio_tmp")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from nltk import word_tokenize
from omegaconf import OmegaConf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from skimage import io
from sklearn.metrics import classification_report, accuracy_score
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

from model.lavis.processors import BaseProcessor
from model.lavis.common.registry import registry
from model.lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from model.lavis.datasets.datasets.base_dataset import BaseDataset
from model.lavis.datasets.datasets.caption_datasets import __DisplMixin

from transformers import CLIPProcessor


@registry.register_processor("my_blip_caption")
class MyBlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([!\"()*#;~])",
            " ",
            caption,
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }

class MyReportProcessor():
    def __init__(self, prompt="", max_words=50, prompt_neg=""):
        self.prompt = prompt
        self.max_words = max_words
        self.prompt_neg = prompt_neg

    def __call__(self, findings, no_labels=False):
        prompt = self.prompt

        if no_labels:
            findings = "no common findings"  # cannot write which findings as we don't no them
        prompt = prompt.format(findings=findings)

        return prompt

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)


class MIMIC_CXR_Dataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, split, cfg, ann_paths=[], truncate=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Preprocessed CSVs already contain: subject_id, study_id, dicom_id, split,
        # ViewPosition, image_path, findings_clean, impression_clean,
        # findings_word_count, has_chexpert_label.
        # Produced by preporcessing/preprocess_mimic_cxr.py over the full p10-p19 set.
        # NOTE: as of 2026-07-20 the splits are NOT view-filtered — they keep every
        # ViewPosition (AP/PA/LATERAL/LL/UNKNOWN) so multi-view fusion can use them.
        # Filter on ViewPosition here if a run needs frontal-only.
        self.cur_split = split
        csv_map = {
            "train": PROCESSED_TRAIN_CSV,
            "val":   PROCESSED_VAL_CSV,
            "test":  PROCESSED_TEST_CSV,
        }
        if split not in csv_map:
            raise ValueError(f"Unknown split '{split}' (expected train/val/test)")
        self.reports = pd.read_csv(csv_map[split])
        self.reports = self.reports.dropna(subset=['findings_clean'])
        self.reports = self.reports.rename(columns={'findings_clean': 'findings'})

        self.use_pred_labels = True

        self.chexpert = pd.read_csv(CHEXPERT_CSV)

        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

        ### This is with uncertain ###
        # #replace uncertain(-1) label with 2. CE loss do not accept negative labels
        self.chexpert[self.chexpert_cols] = self.chexpert[self.chexpert_cols].replace(-1, 2)
        
        for column in  self.chexpert_cols:
            if column != 'No Finding':
                self.chexpert[column].fillna(0.0, inplace=True) # fill with negative class for NaN values

            # Fill NaN values in 'no_finding' with 0.0 (negative)
            self.chexpert['No Finding'].fillna(0.0, inplace=True)
        
        print(f"Number of chexpert records: {len(self.chexpert)}")
         ### fo positve and negative classes       
        #replace uncertain(-1) label with negative. CE loss do not accept negative labels
        # self.chexpert[self.chexpert_cols] = self.chexpert[self.chexpert_cols].replace(-1, 0.0)
        
        # for column in  self.chexpert_cols:
        #     self.chexpert[column].fillna(0.0, inplace=True) # fill with uncertain class for NaN values

        self.custom_epochs_per_epoch = 2 if split == 'train' and cfg.run_cfg.task != "image_text_pretrain_eval" and truncate==None else 1
        self.current_custom_epoch = 0
        self.vit_model = cfg.model_cfg['vit_model']
        self.vit_model_cls = cfg.model_cfg['vit_model_cls']
        self.img_size = cfg.datasets_cfg.mimic_cxr.vis_processor.train.image_size  # should be 224 for coco models, 448 for biovil models
        self.general_trans = transforms.Compose([Resize((512, 512)), CenterCrop(448), ToTensor(), ExpandChannels()])
        if split == 'train' or split == 'val':
            self.vis_augs = transforms.Compose([transforms.RandomAffine(degrees=30, shear=15),
                                        transforms.ColorJitter(brightness=0.2, contrast=0.2)])
            
        else:
            self.vis_augs = None
            
        if self.vit_model == 'biovil':
            self.vis_transforms = create_chest_xray_transform_for_inference(512, center_crop_size=self.img_size)
        
        # for model in self.vit_model_cls:
        #     if model == 'pubmedclip':
        #         self.pubmed_processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        
                
        self.img_ids = {img_id: i for i, img_id in enumerate(self.reports['dicom_id'])}
        self.id_to_dicom = {v: k for k, v in self.img_ids.items()}

        # Split + PA/AP filtering already done at preprocessing time.
        self.annotation = self.reports
        if truncate is not None:
            self.annotation = self.annotation[:truncate]

        print(f"Number of annotation records: {len(self.annotation)}")

        self.annotation = self.annotation.copy()
        self.annotation['findings'] = self.annotation['findings'].apply(lambda x: x.replace('\n', ''))

        # subject_id / study_id are already typed int in the preprocessed CSV.
        self.annotation = pd.merge(self.annotation, self.chexpert, how='inner', on='study_id')
        
        print(f"Number of annotation records: {len(self.annotation)}")
        
        self.annotation =  self.annotation.reset_index()
        
        if self.cur_split == 'train' and truncate is None:
            print(f"{self.cur_split} annotation saved")
            self.annotation.to_csv("train_annotation.csv", index=False)
        
        add_findings_in_prompt = cfg.run_cfg.get("add_findings_in_prompt", False)
        self.prompt = cfg.datasets_cfg.mimic_cxr.text_processor.train.prompt if split == 'train' \
            else cfg.datasets_cfg.mimic_cxr.text_processor.eval.prompt

        self.text_processor = MyReportProcessor(
            prompt=self.prompt, max_words=1000)

        self.evaluator = MIMICEvalCap(self.annotation, self.img_ids)

        # Optional precomputed frozen-encoder feature cache.
        self._init_feature_cache(cfg)

        # Optional multi-view study grouping. When off, nothing below runs and
        # the dataset behaves exactly as the single-view original.
        self._init_study_index(cfg)

    VIEW_ID_MAP = {"PA": 0, "AP": 1, "LATERAL": 2, "LL": 2}
    UNKNOWN_VIEW_ID = 3

    def _view_id(self, view_position):
        if view_position is None or (isinstance(view_position, float) and np.isnan(view_position)):
            return self.UNKNOWN_VIEW_ID
        return self.VIEW_ID_MAP.get(str(view_position).strip().upper(), self.UNKNOWN_VIEW_ID)

    def _init_study_index(self, cfg):
        """Group rows by ``study_id`` into anchor + auxiliary views.

        Every study keeps exactly one anchor row, and all sample keys that
        existed before (``text_output``, ``image_id``, ``classification_labels``,
        ``dicom_id``, ``image_path``, ``image``) continue to come from that
        anchor row, so ``MIMICEvalCap`` keeps working untouched.

        Degrades gracefully when the CSV has no ``ViewPosition`` column: every
        view maps to ``unknown`` and the anchor falls back to row order.
        """
        self.multi_view = bool(cfg.model_cfg.get("multi_view", False))
        self.studies = None
        if not self.multi_view:
            return

        data_cfg = cfg.model_cfg.get("data", {}) or {}
        self.max_aux_views = int(data_cfg.get("max_aux_views", 1))
        anchor_priority = list(data_cfg.get("anchor_priority", ["PA", "AP", "lateral"]))
        # Priority rank per view id; anything unlisted sorts last.
        rank = {}
        for i, name in enumerate(anchor_priority):
            rank[self._view_id(name)] = i
        default_rank = len(anchor_priority)

        has_view_col = "ViewPosition" in self.annotation.columns
        if not has_view_col:
            print(f"[{self.cur_split}] multi_view on but CSV has no ViewPosition "
                  f"column; all views treated as 'unknown'")

        view_ids = (
            self.annotation["ViewPosition"].map(self._view_id).tolist()
            if has_view_col
            else [self.UNKNOWN_VIEW_ID] * len(self.annotation)
        )

        groups = {}
        for pos, study_id in enumerate(self.annotation["study_id"].tolist()):
            groups.setdefault(study_id, []).append(pos)

        self.studies = []
        for study_id, positions in groups.items():
            # Stable sort on priority keeps the first row among equals.
            ordered = sorted(
                positions,
                key=lambda p: rank.get(view_ids[p], default_rank),
            )
            anchor = ordered[0]
            aux = ordered[1:1 + self.max_aux_views]
            self.studies.append({
                "anchor": anchor,
                "aux": aux,
                "anchor_view_id": view_ids[anchor],
                "aux_view_ids": [view_ids[p] for p in aux],
            })

        n_multi = sum(1 for s in self.studies if s["aux"])
        print(f"[{self.cur_split}] multi-view: {len(self.studies)} studies from "
              f"{len(self.annotation)} rows, {n_multi} with >=1 auxiliary view "
              f"(max_aux_views={self.max_aux_views})")

    def _init_feature_cache(self, cfg):
        """Open per-encoder feature memmaps if ``run.feature_cache_dir`` is set.

        Cache layout (see pretraining/precompute_features.py):
            <dir>/<encoder>/<split>_feats.npy   (memmap (N, P, D) float16)
            <dir>/<encoder>/<split>_ids.json    (list[dicom_id] in row order)
        When active, ``__getitem__`` returns the cached raw features instead of
        decoding the JPG, and the model skips the frozen encoder forward.
        """
        self.feature_cache = None
        cache_dir = cfg.run_cfg.get("feature_cache_dir", None)
        if not cache_dir:
            return
        encoders = cfg.model_cfg.get("encoders", {})
        wanted = [e for e in ("biovil", "swin", "raddino") if encoders.get(e, False)]
        cache = {}
        for enc in wanted:
            feats_path = os.path.join(cache_dir, enc, f"{self.cur_split}_feats.npy")
            ids_path = os.path.join(cache_dir, enc, f"{self.cur_split}_ids.json")
            if not (os.path.exists(feats_path) and os.path.exists(ids_path)):
                raise FileNotFoundError(
                    f"feature_cache_dir set but cache missing for encoder '{enc}', "
                    f"split '{self.cur_split}': {feats_path}"
                )
            with open(ids_path) as f:
                row_ids = json.load(f)
            cache[enc] = {
                "feats": np.load(feats_path, mmap_mode="r"),
                "row": {dicom: i for i, dicom in enumerate(row_ids)},
            }
        self.feature_cache = cache
        print(f"[{self.cur_split}] feature cache active for {wanted} from {cache_dir}")

    def set_custom_epoch(self, custom_epoch):
        self.current_custom_epoch = custom_epoch

    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")


    def _row_visual(self, ann):
        """Visual input for one CSV row: decoded image, or cached raw features."""
        # CSV stores absolute Kaggle path (e.g. /kaggle/input/datasets/<slug>/mimic-cxr-jpg-lite/p10/...).
        # Re-anchor to the current vis_root so the same code runs locally and on Kaggle.
        raw = ann["image_path"].replace("\\", "/")
        marker = "/mimic-cxr-jpg-lite/"
        rel = raw.split(marker, 1)[-1] if marker in raw else raw.lstrip("/")
        image_path = os.path.join(self.vis_root, rel)

        out = {"image_path": str(image_path)}
        if self.feature_cache is None:
            out["image"] = self.general_trans(self.load_image(Path(image_path)))
        else:
            for enc, store in self.feature_cache.items():
                row = store["row"][ann["dicom_id"]]
                out[f"{enc}_feat"] = torch.from_numpy(
                    np.ascontiguousarray(store["feats"][row])
                ).float()
        return out

    def __getitem__(self, index):
        start_time = time()
        if self.multi_view:
            subset_size = len(self.studies) // self.custom_epochs_per_epoch
            study = self.studies[self.current_custom_epoch * subset_size + index]
            ann = self.annotation.iloc[study["anchor"]]
        else:
            study = None
            subset_size = len(self.annotation) // self.custom_epochs_per_epoch
            start_index = self.current_custom_epoch * subset_size
            actual_index = start_index + index
            ann = self.annotation.iloc[actual_index]

        anchor_visual = self._row_visual(ann)
        image_path = anchor_visual.pop("image_path")

        # if self.cur_split == 'train':
        #     if self.vit_model == "biovil":  # old version worked with smaller img and without biovil img processing
        #         image_biovil = self.vis_transforms(image)
        #         if self.vis_augs is not None:
        #             image_biovil = self.vis_augs(image_biovil)

        #         # print(f"image_biovil shape is {image_biovil.shape}")
                
        #     for model in self.vit_model_cls:
        #         if model == 'pubmedclip':
        #             image_pubmed = self.general_trans(image)
        #             image_pubmed = self.pubmed_processor(images=image_pubmed, return_tensors="pt")
        #             image_pubmed = image_pubmed['pixel_values'].squeeze(0)
        #             if self.vis_augs is not None:
        #                 image_pubmed = self.vis_augs(image_pubmed)
                
                
                # print(f"image pubmed shape is {image_pubmed.shape}")
                 
                 
        caption = ann["findings"].strip()
        # input_text = self.text_processor(findings=None)

        # conv = Conversation(
        #     system="A chat between a curious user and an artificial intelligence assistant acting as an experienced radiologist. "
        #            "The assistant gives professional, detailed, and polite answers to the user's questions.",
        #     roles=["USER", "ASSISTANT"],
        #     messages=[],
        #     offset=0,
        #     sep_style=SeparatorStyle.TWO,
        #     sep=" ",
        #     sep2="</s>",
        # )
        # conv.append_message(conv.roles[0], input_text)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # if "<IMG>" in prompt:
        #     before_img, after_img = prompt.split("<IMG>")
        #     prompt = (before_img, after_img)


        # Get the CheXpert classification labels (e.g., 0 for negative, 1 for positive, -1 for uncertain)
        chexpert_labels = ann[self.chexpert_cols].values.astype(float)
        # print(torch.tensor(chexpert_labels, dtype=torch.long))
        
        end_time = time()
        # print(f"__getitem__ took {end_time - start_time:.4f} seconds")
        
        sample = {
            # "text_input": prompt,
            "text_output": caption,
            "image_id": self.img_ids[ann["dicom_id"]],
            "classification_labels": torch.tensor(chexpert_labels, dtype=torch.long),  # Convert to tensor
            "dicom_id": ann["dicom_id"],
            "image_path": str(image_path)
        }
        sample.update(anchor_visual)  # "image", or the cached "<enc>_feat" tensors

        if study is not None:
            aux_visuals = [
                self._row_visual(self.annotation.iloc[p]) for p in study["aux"]
            ]
            sample["anchor_view_id"] = study["anchor_view_id"]
            sample["aux_view_ids"] = list(study["aux_view_ids"])
            if self.feature_cache is None:
                sample["aux_image"] = [a["image"] for a in aux_visuals]
            else:
                for enc in self.feature_cache:
                    sample[f"aux_{enc}_feat"] = [a[f"{enc}_feat"] for a in aux_visuals]
        return sample

    def __len__(self):
        if self.multi_view:
            return len(self.studies) // self.custom_epochs_per_epoch
        return len(self.annotation) // self.custom_epochs_per_epoch

    def collater(self, samples):
        """Pad ragged auxiliary-view counts to the batch's N_max.

        Pre-existing keys are delegated to the default collate untouched, so a
        ``multi_view=False`` batch is byte-identical to the original.
        """
        if not self.multi_view:
            return super().collater(samples)

        aux_keys = [
            k for k in samples[0]
            if k == "aux_image" or (k.startswith("aux_") and k.endswith("_feat"))
        ]
        skip = set(aux_keys) | {"aux_view_ids"}
        batch = super().collater(
            [{k: v for k, v in s.items() if k not in skip} for s in samples]
        )

        B = len(samples)
        n_max = max(len(s["aux_view_ids"]) for s in samples)
        aux_mask = torch.zeros(B, n_max, dtype=torch.bool)
        aux_view_ids = torch.full(
            (B, n_max), self.UNKNOWN_VIEW_ID, dtype=torch.long
        )
        for i, s in enumerate(samples):
            n = len(s["aux_view_ids"])
            if n:
                aux_mask[i, :n] = True
                aux_view_ids[i, :n] = torch.tensor(s["aux_view_ids"], dtype=torch.long)
        batch["aux_mask"] = aux_mask
        batch["aux_view_ids"] = aux_view_ids

        for key in aux_keys:
            anchor_key = "image" if key == "aux_image" else key[len("aux_"):]
            template = samples[0][anchor_key]
            if n_max == 0:
                batch[key] = torch.zeros(
                    (B, 0) + tuple(template.shape), dtype=template.dtype
                )
                continue
            rows = []
            for s in samples:
                items = list(s[key])
                items += [torch.zeros_like(s[anchor_key])] * (n_max - len(items))
                rows.append(torch.stack(items, dim=0))
            batch[key] = torch.stack(rows, dim=0)
        return batch


@registry.register_builder("mimic_cxr")
class MIMIC_CXR_Builder(BaseDatasetBuilder):
    train_dataset_cls = MIMIC_CXR_Dataset
    eval_dataset_cls = MIMIC_CXR_Dataset

    DATASET_CONFIG_DICT = {
        "default": "defaults_report.yaml"
    }


class MIMICEvalCap:
    def __init__(self, gts, img_id_map):

        self.gts = gts

        # invert img_id_map
        self.dicom_to_id = img_id_map
        self.id_to_dicom = {v: k for k, v in img_id_map.items()}

        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L")
        ]


    def preprocess(self, s):
        s = s.replace('\n', '')
        s = s.replace('<s>', '')
        s = s.replace('</s>', '')
        return s

    def evaluate(self, res):

        res = {self.id_to_dicom[elem["image_id"]]: elem["caption"] for elem in res}
        res_keys_set = set(res.keys())
        gts = {}
        gts_img_id = {}
        for _, elem in self.gts.iterrows():
            dicom_id = elem["dicom_id"]
            if dicom_id in res_keys_set:
                gts[dicom_id] = [elem["findings"]]
                gts_img_id[self.dicom_to_id[dicom_id]] = [elem["findings"]]

        # gts = {elem["dicom_id"]: [elem["findings"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        # gts_img_id = {self.dicom_to_id[elem["findings"]]: [elem["Note"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        assert res.keys() == gts.keys()
        # =================================================
        # Pre-process sentences
        # =================================================
        print('tokenization...')
        for dicom in res.keys():
            pred_text = ' '.join(word_tokenize(self.preprocess(res[dicom]))).lower()
            true_text = ' '.join(word_tokenize(self.preprocess(gts[dicom][0]))).lower()

            res[dicom] = [pred_text]
            gts[dicom] = [true_text]

        # =================================================
        # Compute scores
        # =================================================
        final_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    final_scores[m] = sc
                    #final_scores["elem_wise_" + str(m)] = scs
                    print("%s: %0.3f" % (m, sc))
            else:
                print("%s: %0.3f" % (method, score))
                #final_scores["elem_wise_" + str(method)] = scores
                final_scores[method] = score

        final_scores['agg_metrics'] = np.mean(list({k: v for k, v in final_scores.items() if "elem_wise" not in k}.values()))

        return final_scores, gts_img_id

class CheXpertDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, split, cfg, ann_paths = [], truncate=None):
        """
        vis_root (string): Root directory of images
        ann_path (string): Path to the CheXpert annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Load annotation file
        self.cur_split = split
        self.annotations = pd.read_csv("/workspace/CheXpert-v1.0-small/valid.csv")
        self.custom_epochs_per_epoch = 1
        self.current_custom_epoch = 0
        
        # Filter only frontal images with AP/PA views
        self.annotations = self.annotations[(self.annotations['Frontal/Lateral'] == 'Frontal')]

        # Define the CheXpert label columns
        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
                              "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
                              "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

        # Handle uncertain labels (-1) by replacing with 2 (for cross-entropy loss compatibility)
        # self.annotations[self.chexpert_cols] = self.annotations[self.chexpert_cols].replace(-1, 2)
        
        for column in self.chexpert_cols:
            self.annotations[column].fillna(0.0, inplace=True)

        # Define transformations
        self.img_size = cfg.datasets_cfg.mimic_cxr.vis_processor.train.image_size
        self.general_trans = transforms.Compose([Resize((512, 512)), CenterCrop(448), ToTensor(), ExpandChannels()])
        self.img_ids = {path: i for i, path in enumerate(self.annotations['Path'])}
        if truncate is not None:
            self.annotations = self.annotations[:truncate]

        print(f"Number of annotation records: {len(self.annotations)}")
        
    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")

    def set_custom_epoch(self, custom_epoch):
        self.current_custom_epoch = custom_epoch

    def __getitem__(self, index):
        start_time = time()
        subset_size = len(self.annotations) // self.custom_epochs_per_epoch
        start_index = self.current_custom_epoch * subset_size
        actual_index = start_index + index

        ann = self.annotations.iloc[actual_index]

        image_path = os.path.join(self.vis_root, ann["Path"])
        image = self.load_image(Path(image_path))
        image = self.general_trans(image)
        
        
        chexpert_labels = ann[self.chexpert_cols].values.astype(float)
        
        end_time = time()
        return {
            "image": image,
            "image_id": self.img_ids[ann["Path"]],
            "classification_labels": torch.tensor(chexpert_labels, dtype=torch.long),
            "image_path": str(image_path)
        }
    
    def __len__(self):
        return len(self.annotations)
    

class IU_Xray_Dataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, split, cfg, ann_paths=[], truncate=None):
        """
        Args:
            vis_processor: Vision processor
            text_processor: Text processor  
            vis_root (string): Root directory with all the images and reports
            split (string): 'train', 'val', or 'test'
            cfg: Configuration
            ann_paths: Annotation paths
            truncate: Optional truncation
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        # Load reports
        self.reports_df = pd.read_csv(os.path.join(vis_root, 'indiana_reports.csv'))
        
        # Load projections
        self.projections_df = pd.read_csv(os.path.join(vis_root, 'indiana_projections.csv'))
        
        # Filter to only include frontal images
        self.frontal_images = self.projections_df[self.projections_df['projection'] == 'Frontal']
        
        # Create train/val/test split
        total_cases = len(self.frontal_images)
        train_size = int(0.7 * total_cases)
        val_size = int(0.2 * total_cases)
        
        if split == 'train':
            self.annotations = self.frontal_images[:train_size]
        elif split == 'val':
            self.annotations = self.frontal_images[train_size:train_size + val_size]
        elif split == 'test':
            self.annotations = self.frontal_images[train_size + val_size:]
            
        # Create mapping from uid to report text
        self.uid_to_report = dict(zip(self.reports_df['uid'], self.reports_df['findings']))
        
        self.img_size = cfg.datasets_cfg.mimic_cxr.vis_processor.train.image_size
        self.general_trans = transforms.Compose([Resize((512, 512)), CenterCrop(448), ToTensor(), ExpandChannels()])
        # Create image id mapping
        self.img_ids = {row['filename']: idx for idx, row in self.annotations.iterrows()}
        self.id_to_img = {v: k for k, v in self.img_ids.items()}
        
        # Store split
        self.split = split

        if truncate is not None:
            self.annotations = self.annotations[:truncate]
    
    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.vis_root, 'images', 'images_normalized', ann['filename'])
        image = self.load_image(Path(image_path))
        image = self.general_trans(image)
            
        # Get report text
        report = self.uid_to_report.get(ann['uid'], '')
        
        # return {
        #     'image': image,
        #     'image_id': self.img_ids[ann['filename']],
        #     'text_output': report,
        #     'image_path': str(image_path)
        # }
        return {
            'image': image,
            'image_id': self.img_ids[ann['filename']]
        }
