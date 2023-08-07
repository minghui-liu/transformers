# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Vit3d."""

import warnings

from ...utils import logging
from .image_processing_vit3d import Vit3dImageProcessor


logger = logging.get_logger(__name__)


class Vit3dFeatureExtractor(Vit3dImageProcessor):
    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "The class Vit3dFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please"
            " use Vit3dImageProcessor instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)
