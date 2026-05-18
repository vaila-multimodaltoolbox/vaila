---
license: other
extra_gated_fields:
  First Name: text
  Last Name: text
  Date of birth: date_picker
  Country: country
  Affiliation: text
  Job title:
    type: select
    options:
    - Student
    - Research Graduate
    - AI researcher
    - AI developer/engineer
    - Reporter
    - Other
  geo: ip_location
  By clicking Submit below I accept the terms of the license and acknowledge that the information I provide will be collected stored processed and shared in accordance with the Meta Privacy Policy: checkbox
extra_gated_description: >-
  The information you provide will be collected, stored, processed and shared in
  accordance with the [Meta Privacy
  Policy](https://www.facebook.com/privacy/policy/).
extra_gated_button_content: Submit
language:
- en
pipeline_tag: mask-generation
library_name: checkpoint
tags:
- sam3.1
---
# SAM 3.1
[SAM 3](https://github.com/facebookresearch/sam3) (Segment Anything with Concepts) is a unified foundation model from Meta for promptable segmentation in images and videos. It detects, segments, and tracks objects using text or visual prompts such as points, boxes, and masks. SAM 3 introduces the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase, handling over 50x more unique concepts than existing benchmarks. SAM 3.1 builds on this with **Object Multiplex**, a shared-memory approach for joint multi-object tracking that delivers ~7x faster inference at 128 objects on a single H100 GPU without sacrificing accuracy, along with improved VOS performance on 6 out of 7 benchmarks.

<p align="center">
  <img src="https://github.com/facebookresearch/sam3/blob/main/assets/sam3.1_diagram.png?raw=true" width="720" />
</p>

This repository hosts only the SAM 3.1 model checkpoints — there is no Hugging Face Transformers integration. For installation, code, usage examples, and full documentation, please visit the [SAM 3 GitHub repository](https://github.com/facebookresearch/sam3).
