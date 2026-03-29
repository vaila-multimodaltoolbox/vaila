---
license: other
license_name: sam-license
license_link: https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE
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
tags:
- sam-3d
- sam-3d-body
- 3d-human-mesh-recovery
- human-pose-estimation
- meta
library_name: sam-3d-body
---

# SAM 3D Body: Robust Full-Body Human Mesh Recovery

**Meta Superintelligence Labs**

**SAM 3D Body (3DB)** is a promptable model for single-image full-body 3D human mesh recovery (HMR). Our method demonstrates state-of-the-art performance, with strong generalization and consistent accuracy in diverse in-the-wild conditions. 3DB estimates the human pose of the body, feet, and hands based on the [Momentum Human Rig](https://github.com/facebookresearch/MHR) (MHR), a new parametric mesh representation that decouples skeletal structure and surface shape for improved accuracy and interpretability.

3DB employs an encoder-decoder architecture and supports auxiliary prompts, including 2D keypoints and masks, enabling user-guided inference similar to the SAM family of models. Our model is trained on high-quality annotations from a multi-stage annotation pipeline using differentiable optimization, multi-view geometry, dense keypoint detection, and a data engine to collect and annotated data covering both common and rare poses across a wide range of viewpoints.

## Key Features

- **Robust Full-Body Performance**: Superior handling of occlusions, hard poses, and challenging viewpoints
- **Promptable Model**: Supports auxiliary prompts including 2D keypoints and masks for user-guided inference
- **Momentum Human Rig (MHR)**: New parametric mesh representation that decouples skeletal structure and surface shape
- **Large-Scale and High-Quality Data**: Multi-stage annotation pipeline for large-scale, diverse, and high-quality data

## Quick Start

### Installation

Please refer to [INSTALL.md](https://github.com/facebookresearch/sam-3d-body/blob/main/INSTALL.md) for detailed installation guidelines.

### Inference

```bash
# Download assets from HuggingFace
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3

# Run demo script
python demo.py \
    --image_folder <path_to_images> \
    --output_folder <path_to_output> \
    --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
    --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

You can also try the following lines of code with models loaded directly from [Hugging Face](https://huggingface.co/facebook)

```python
import cv2
import numpy as np
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

# Set up the estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")

# Load and process image
img_bgr = cv2.imread("path/to/image.jpg")
outputs = estimator.process_one_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Visualize and save results
rend_img = visualize_sample_together(img_bgr, outputs, estimator.faces)
cv2.imwrite("output.jpg", rend_img.astype(np.uint8))
```

For a complete demo with visualization, see [demo_human.ipynb](https://github.com/facebookresearch/sam-3d-body/blob/main/notebook/demo_human.ipynb).

## Output Format

Each detected person returns a dictionary containing:

- `pred_vertices`: 3D mesh vertices in camera coordinates
- `pred_keypoints_3d`: 3D pose keypoints
- `pred_keypoints_2d`: 2D pose keypoints projected to image
- `pred_cam_t`: Camera translation parameters
- `focal_length`: Estimated focal length
- `body_pose_params`: Body pose parameters
- `hand_pose_params`: Hand pose parameters
- `shape_params`: Body shape parameters


## Citation

If you use SAM 3D Body or the SAM 3D Body dataset in your research, please use the following BibTeX entry.

```bibtex
@article{yang2025sam3dbody,
  title={SAM 3D Body: Robust Full-Body Human Mesh Recovery},
  author={Yang, Xitong and Kukreja, Devansh and Pinkus, Don and Sagar, Anushka and Fan, Taosha and Park, Jinhyung and Shin, Soyong and Cao, Jinkun and Liu, Jiawei and Ugrinovic, Nicolas and Feiszli, Matt and Malik, Jitendra and Dollar, Piotr and Kitani, Kris},
  journal={arXiv preprint; identifier to be added},
  year={2025}
}
```

## License

The SAM 3D Body model is licensed under [SAM License](https://huggingface.co/facebook/sam-3d-body-dinov3/blob/main/LICENSE).

## Links

- **Paper**: [https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/](https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/)
- **Code**: [https://github.com/facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body)
- **Demo**: [https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d](https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d)
- **Website**: [https://ai.meta.com/sam3d/](https://ai.meta.com/sam3d/)
- **Dataset**: [https://huggingface.co/datasets/facebook/sam-3d-body-dataset](https://huggingface.co/datasets/facebook/sam-3d-body-dataset)