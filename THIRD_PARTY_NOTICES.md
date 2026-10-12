# Third-party notices

## TrackNetV3 (shuttle detection)

`backend/rallycv/tracknet.py` contains a re-implementation of the TrackNet
network from https://github.com/qaz812345/TrackNetV3, and the pipeline loads
the authors' pretrained checkpoint (downloaded by
`backend/scripts/download_models.py`).

    MIT License

    Copyright (c) 2024 qaz812345

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Paper: Chen, Y.-J., & Wang, Y.-S. (2023). TrackNetV3: Enhancing ShuttleCock
Tracking with Augmentations and Trajectory Rectification.

## RT-DETRv2 (person detection) — Apache-2.0

Player detection uses RT-DETRv2 (`PekingU/rtdetr_v2_r18vd` by default, set with
`RALLYREVIEW_PERSON_MODEL`) through Hugging Face Transformers, both licensed
under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
The weights are downloaded from the Hugging Face Hub by
`backend/scripts/download_models.py`. If you switch to another model, check
its licence.

Paper: Lv, W., et al. (2024). RT-DETRv2: Improved Baseline with Bag-of-Freebies
for Real-Time Detection Transformer.

## Sample footage (not distributed)

`backend/data/samples/` (git-ignored) holds clips used for manual testing,
downloaded from Wikimedia Commons under CC BY 3.0:
"Wang Zhiyi triumph in all-Chinese China Open finals I China Open 2025".
