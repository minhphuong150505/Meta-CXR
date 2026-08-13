"""CPU-only test bootstrap.

`model/lavis/__init__.py` eagerly imports the dataset builders, which drag in the
whole training stack (torchvision, transformers, opencv, decord). The tests here
only touch `optims.py` and `base_task.py`, so register `model` and `model.lavis`
as path-only packages first: Python then resolves the submodules without ever
executing that `__init__`.

`dist_utils` also imports `timm.models.hub` for a checkpoint-download helper that
no test calls, so stub it when timm is absent.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _name in ("model", "model.lavis"):
    if _name not in sys.modules:
        _pkg = types.ModuleType(_name)
        _pkg.__path__ = [str(_REPO_ROOT / Path(*_name.split(".")))]
        sys.modules[_name] = _pkg

try:  # pragma: no cover - exercised only on machines without timm
    import timm.models.hub  # noqa: F401
except ImportError:
    _timm = types.ModuleType("timm")
    _models = types.ModuleType("timm.models")
    _hub = types.ModuleType("timm.models.hub")
    _hub.get_cache_dir = lambda *args, **kwargs: "/tmp"
    _hub.download_cached_file = lambda *args, **kwargs: None
    _models.hub = _hub
    _timm.models = _models
    sys.modules.update({"timm": _timm, "timm.models": _models, "timm.models.hub": _hub})


@pytest.fixture
def report_dataset_module(monkeypatch):
    """Load ReportDataset with narrow CPU stubs for its optional training stack.

    The fixture is opt-in, so the known baseline failures caused by missing
    torchvision/transformers remain unchanged.  Its Pillow transform stubs cover
    only the paths exercised by ``test_explanation_mask_pipeline.py``.
    """

    import numpy as np
    import torch
    from PIL import Image

    class Compose:
        def __init__(self, operations):
            self.operations = list(operations)

        def __call__(self, value):
            for operation in self.operations:
                value = operation(value)
            return value

    class Resize:
        def __init__(self, size):
            self.size = size

        def __call__(self, image):
            if isinstance(self.size, int):
                width, height = image.size
                if width <= height:
                    output_size = (self.size, int(self.size * height / width))
                else:
                    output_size = (int(self.size * width / height), self.size)
            else:
                output_size = (int(self.size[1]), int(self.size[0]))
            return image.resize(output_size, resample=Image.Resampling.BILINEAR)

    class CenterCrop:
        def __init__(self, size):
            self.size = (size, size) if isinstance(size, int) else tuple(size)

        def __call__(self, image):
            crop_height, crop_width = self.size
            width, height = image.size
            left = int(round((width - crop_width) / 2.0))
            top = int(round((height - crop_height) / 2.0))
            return image.crop((left, top, left + crop_width, top + crop_height))

    class ToTensor:
        def __call__(self, image):
            array = np.array(image, dtype=np.float32, copy=True)
            if array.ndim == 2:
                array = array[None]
            else:
                array = np.transpose(array, (2, 0, 1))
            return torch.from_numpy(array) / 255.0

    class RandomApply:
        def __init__(self, operations, p=0.5):
            self.operations = list(operations)
            self.p = float(p)

        def __call__(self, value):
            if torch.rand(1).item() >= self.p:
                return value
            for operation in self.operations:
                value = operation(value)
            return value

    class InterpolationMode:
        NEAREST = Image.Resampling.NEAREST
        BILINEAR = Image.Resampling.BILINEAR

    functional = types.ModuleType("torchvision.transforms.functional")

    def affine(
        image,
        *,
        angle,
        translate,
        scale,
        shear,
        interpolation,
        fill,
    ):
        if float(angle) != 0.0 or float(scale) != 1.0 or any(float(v) != 0.0 for v in shear):
            raise AssertionError("CPU torchvision stub only supports translated test fixtures")
        x_shift, y_shift = translate
        return image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1.0, 0.0, -float(x_shift), 0.0, 1.0, -float(y_shift)),
            resample=interpolation,
            fillcolor=fill,
        )

    functional.affine = affine

    class RandomAffine:
        fixed_params = (0.0, (0, 0), 1.0, (0.0, 0.0))
        get_params_calls = 0

        def __init__(self, degrees, translate=None, scale=None, shear=None):
            self.degrees = degrees
            self.translate = translate
            self.scale = scale
            self.shear = shear

        @staticmethod
        def get_params(degrees, translate, scale_ranges, shears, img_size):
            del degrees, translate, scale_ranges, shears, img_size
            RandomAffine.get_params_calls += 1
            return RandomAffine.fixed_params

        def __call__(self, image):
            angle, translate, scale, shear = self.get_params(
                self.degrees, self.translate, self.scale, self.shear, list(image.size)
            )
            return affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

    class ColorJitter:
        def __init__(self, **kwargs):
            del kwargs

        def __call__(self, image):
            return image

    transforms = types.ModuleType("torchvision.transforms")
    transforms.Compose = Compose
    transforms.Resize = Resize
    transforms.CenterCrop = CenterCrop
    transforms.ToTensor = ToTensor
    transforms.RandomApply = RandomApply
    transforms.RandomAffine = RandomAffine
    transforms.ColorJitter = ColorJitter
    transforms.InterpolationMode = InterpolationMode
    transforms.functional = functional
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = transforms
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "torchvision.transforms", transforms)
    monkeypatch.setitem(sys.modules, "torchvision.transforms.functional", functional)

    local_config = types.ModuleType("local_config")
    for name in (
        "JAVA_HOME",
        "JAVA_PATH",
        "SPLIT_CSV",
        "REPORTS_CSV",
        "CHEXPERT_CSV",
        "METADATA_CSV",
        "PROCESSED_TRAIN_CSV",
        "PROCESSED_VAL_CSV",
        "PROCESSED_TEST_CSV",
    ):
        setattr(local_config, name, "")
    monkeypatch.setitem(sys.modules, "local_config", local_config)

    nltk = types.ModuleType("nltk")
    nltk.word_tokenize = lambda value: str(value).split()
    monkeypatch.setitem(sys.modules, "nltk", nltk)

    def package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    package("pycocoevalcap")
    package("pycocoevalcap.bleu")
    package("pycocoevalcap.meteor")
    package("pycocoevalcap.rouge")
    for module_name, class_name in (
        ("pycocoevalcap.bleu.bleu", "Bleu"),
        ("pycocoevalcap.meteor.meteor", "Meteor"),
        ("pycocoevalcap.rouge.rouge", "Rouge"),
    ):
        module = types.ModuleType(module_name)
        setattr(module, class_name, type(class_name, (), {}))
        monkeypatch.setitem(sys.modules, module_name, module)

    skimage = package("skimage")
    skimage_io = types.ModuleType("skimage.io")
    skimage_io.imread = lambda _path: np.zeros((1, 1), dtype=np.uint8)
    skimage.io = skimage_io
    monkeypatch.setitem(sys.modules, "skimage.io", skimage_io)
    package("sklearn")
    sklearn_metrics = types.ModuleType("sklearn.metrics")
    sklearn_metrics.classification_report = lambda *args, **kwargs: {}
    sklearn_metrics.accuracy_score = lambda *args, **kwargs: 0.0
    monkeypatch.setitem(sys.modules, "sklearn.metrics", sklearn_metrics)

    processors = types.ModuleType("model.lavis.processors")
    processors.BaseProcessor = type("BaseProcessor", (), {})
    monkeypatch.setitem(sys.modules, "model.lavis.processors", processors)

    common = package("model.lavis.common")
    registry_module = types.ModuleType("model.lavis.common.registry")

    class DummyRegistry:
        @staticmethod
        def register_processor(_name):
            return lambda value: value

        @staticmethod
        def register_builder(_name):
            return lambda value: value

    registry_module.registry = DummyRegistry()
    common.registry = registry_module
    monkeypatch.setitem(sys.modules, "model.lavis.common.registry", registry_module)

    datasets = package("model.lavis.datasets")
    builders = package("model.lavis.datasets.builders")
    builder_module = types.ModuleType("model.lavis.datasets.builders.base_dataset_builder")
    builder_module.BaseDatasetBuilder = type("BaseDatasetBuilder", (), {})
    builders.base_dataset_builder = builder_module
    monkeypatch.setitem(
        sys.modules, "model.lavis.datasets.builders.base_dataset_builder", builder_module
    )
    dataset_package = package("model.lavis.datasets.datasets")
    base_dataset_module = types.ModuleType("model.lavis.datasets.datasets.base_dataset")

    class BaseDataset:
        def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=None):
            del ann_paths
            self.vis_processor = vis_processor
            self.text_processor = text_processor
            self.vis_root = vis_root

        @staticmethod
        def collater(samples):
            return torch.utils.data.default_collate(samples)

    base_dataset_module.BaseDataset = BaseDataset
    caption_module = types.ModuleType("model.lavis.datasets.datasets.caption_datasets")
    caption_module.__DisplMixin = type("__DisplMixin", (), {})
    dataset_package.base_dataset = base_dataset_module
    dataset_package.caption_datasets = caption_module
    datasets.builders = builders
    datasets.datasets = dataset_package
    monkeypatch.setitem(
        sys.modules, "model.lavis.datasets.datasets.base_dataset", base_dataset_module
    )
    monkeypatch.setitem(
        sys.modules, "model.lavis.datasets.datasets.caption_datasets", caption_module
    )

    module_name = "_report_dataset_explanation_test"
    spec = importlib.util.spec_from_file_location(
        module_name, _REPO_ROOT / "model/lavis/data/ReportDataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._test_random_affine = RandomAffine
    return module
