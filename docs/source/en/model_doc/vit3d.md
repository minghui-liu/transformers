<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# vit3d

## Overview

The vit3d model was proposed in [<INSERT PAPER NAME HERE>](<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## Vit3dConfig

[[autodoc]] Vit3dConfig

## Vit3dFeatureExtractor

[[autodoc]] Vit3dFeatureExtractor
    - __call__


## Vit3dImageProcessor

[[autodoc]] Vit3dImageProcessor
    - preprocess

## Vit3dModel

[[autodoc]] Vit3dModel
    - forward

## Vit3dForMaskedImageModeling

[[autodoc]] Vit3dForMaskedImageModeling
    - forward

## Vit3dForImageClassification

[[autodoc]] Vit3dForImageClassification
    - forward
