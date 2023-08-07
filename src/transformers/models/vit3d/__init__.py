# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)


_import_structure = {"configuration_vit3d": ["VIT3D_PRETRAINED_CONFIG_ARCHIVE_MAP", "Vit3dConfig", "Vit3dOnnxConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_vit3d"] = ["Vit3dFeatureExtractor"]
    _import_structure["image_processing_vit3d"] = ["Vit3dImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vit3d"] = [
        "VIT3D_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Vit3dForImageClassification",
        "Vit3dForMaskedImageModeling",
        "Vit3dModel",
        "Vit3dPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_vit3d import VIT3D_PRETRAINED_CONFIG_ARCHIVE_MAP, Vit3dConfig, Vit3dOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_vit3d import Vit3dFeatureExtractor
        from .image_processing_vit3d import Vit3dImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vit3d import (
            VIT3D_PRETRAINED_MODEL_ARCHIVE_LIST,
            Vit3dForImageClassification,
            Vit3dForMaskedImageModeling,
            Vit3dModel,
            Vit3dPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
