from functools import lru_cache
import numpy as np
import nibabel as nib
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

_TEXT_PROMPTS = [
    "brain MRI scan",
    "medical MRI brain scan",
]


def _extract_middle_slice(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError("Unsupported NIfTI shape")

    z_idx = data.shape[2] // 2
    slice_2d = data[:, :, z_idx].astype(np.float32)

    min_val = float(slice_2d.min())
    max_val = float(slice_2d.max())
    if max_val - min_val < 1e-8:
        raise ValueError("Empty slice")

    norm = (slice_2d - min_val) / (max_val - min_val)
    img_u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_u8).convert("RGB")


@lru_cache(maxsize=2)
def _get_clip(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    return model, processor


def is_valid_mri_input(nifti_path, threshold=0.15):
    try:
        image = _extract_middle_slice(nifti_path)
    except Exception:
        return False

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, processor = _get_clip(device)

        image_inputs = processor(images=image, return_tensors="pt")
        text_inputs = processor(text=_TEXT_PROMPTS, return_tensors="pt", padding=True)

        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)
        max_sim = float(similarity.max().item())
        return max_sim >= threshold
    except Exception:
        return True
