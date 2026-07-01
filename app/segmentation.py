import os
import torch
import nibabel as nib
import numpy as np
import ants
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet

def load_model(model_path, device='cpu'):
    model = SegResNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        init_filters=32,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        dropout_prob=0.2,
    )
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def register_images(fixed_path, moving_paths):
    fixed = ants.image_read(fixed_path)
    registered_images = [fixed]
    for path in moving_paths:
        moving = ants.image_read(path)
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid')
        registered_images.append(reg['warpedmovout'])
    return registered_images

def preprocess_ants(ants_images):
    processed = []
    for img in ants_images:
        data = img.numpy()
        data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        processed.append(data)
    return np.stack(processed, axis=0)

def run_inference(model, input_volume, device='cpu'):
    input_tensor = torch.tensor(input_volume, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            input_tensor,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    pred = torch.argmax(logits, dim=1).squeeze(0)
    mask = pred.cpu().numpy()
    return mask
