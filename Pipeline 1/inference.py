import os
import numpy as np
import nibabel as nib
import torch
import cv2
from tqdm import tqdm
from skimage import morphology

from model import ResNet34UNet
from config import CFG


DEVICE = CFG["DEVICE"]
IMG_SIZE = CFG["IMG_SIZE"]
NUM_CLASSES = CFG["NUM_CLASSES"]


# -----------------------------
# Utils
# -----------------------------
def brats_normalize(ch):
    """Normalize a single channel using percentile clipping."""
    if (ch > 0).sum() > 0:
        p1, p99 = np.percentile(ch[ch > 0], (1, 99))
        ch = np.clip((ch - p1) / (p99 - p1 + 1e-6), 0, 1)
    return ch


def load_slice(patient_path, patient_id, slice_idx):
    """Load and preprocess a single slice from all modalities."""
    images = []
    for mod in ["flair", "t1", "t1ce", "t2"]:
        p = os.path.join(patient_path, f"{patient_id}_{mod}.nii.gz")
        vol = nib.load(p).get_fdata()
        img = vol[..., slice_idx]
        images.append(img)

    img = np.stack(images, axis=-1)

    # resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize
    for c in range(4):
        img[..., c] = brats_normalize(img[..., c])

    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return img


def load_patient_modalities(patient_dir, patient_id):
    """Load all modalities for a patient and return as a dictionary."""
    mods = {}
    for mod in ["flair", "t1", "t1ce", "t2"]:
        p = os.path.join(patient_dir, f"{patient_id}_{mod}.nii.gz")
        if not os.path.exists(p):
            return None
        mods[mod] = nib.load(p).get_fdata()
    return mods


# -----------------------------
# Post-processing
# -----------------------------
def post_process_clean(pred_vol, min_size=50):
    """Remove small connected components from the prediction volume."""
    clean_pred = np.zeros_like(pred_vol)
    for cls in [1, 2, 4]:
        mask = (pred_vol == cls)
        if mask.sum() > 0:
            mask = morphology.remove_small_objects(mask, min_size=min_size)
            clean_pred[mask] = cls
    return clean_pred


# -----------------------------
# Inference on ONE patient
# -----------------------------
def infer_patient(patient_dir, model):
    """Run inference on a single patient using the model."""
    patient_id = os.path.basename(patient_dir)

    seg_path = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")
    proxy = nib.load(seg_path)
    depth = proxy.shape[2]

    pred_volume = np.zeros((IMG_SIZE, IMG_SIZE, depth), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for z in tqdm(range(depth), desc=f"Inference {patient_id}"):
            img = load_slice(patient_dir, patient_id, z).to(DEVICE)

            out = model(img)
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

            pred_volume[..., z] = pred

    return pred_volume


# -----------------------------
# Optimized Inference with Post-processing
# -----------------------------
def predict_patient_optimized(model, patient_dir, pid):
    """Predict tumor segmentation for a patient with post-processing."""
    model.eval()
    mods = load_patient_modalities(patient_dir, pid)
    if mods is None:
        return None

    H, W, D = mods["flair"].shape
    pred_3d = np.zeros((H, W, D), dtype=np.uint8)

    brain_mask = (mods["t1"] > 0) | (mods["flair"] > 0)

    with torch.no_grad():
        for z in range(D):
            slices = []
            for m in ["flair", "t1", "t1ce", "t2"]:
                s = mods[m][:, :, z]
                s = brats_normalize(s)
                s = cv2.resize(s, (IMG_SIZE, IMG_SIZE))
                slices.append(s)

            x = np.stack(slices, axis=0)
            x = torch.from_numpy(x).unsqueeze(0).float().to(DEVICE)

            out = model(x)
            pred = torch.argmax(out.softmax(1), 1).squeeze(0).cpu().numpy()

            pred = cv2.resize(
                pred.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )
            pred_3d[:, :, z] = pred

    pred_3d = pred_3d * brain_mask.astype(np.uint8)
    pred_3d[pred_3d == 3] = 4
    pred_3d = post_process_clean(pred_3d, min_size=30)

    return pred_3d


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    MODEL_PATH = "/kaggle/input/brats-model/resnet34_unet_best.pth"
    PATIENT_PATH = "/kaggle/input/brain-tumor-segmentation-hackathon/BraTS2021_00000"

    model = ResNet34UNet(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Standard inference
    pred_3d = infer_patient(PATIENT_PATH, model)
    print("✅ Inference done")
    print("Prediction shape:", pred_3d.shape)

    # Optimized inference with post-processing
    patient_id = os.path.basename(PATIENT_PATH)
    pred_optimized = predict_patient_optimized(model, PATIENT_PATH, patient_id)
    if pred_optimized is not None:
        print("✅ Optimized inference done")
        print("Optimized prediction shape:", pred_optimized.shape)
