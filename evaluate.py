import os
import torch
import numpy as np
from PIL import Image
import cv2
import yaml
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from torchvision import transforms
from model.model import HIFI
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# ========== 1. Path Configuration ==========
CONFIG_PATH = 'config\config.yaml'
MODEL_PATH = 'bestModels\\best_HIFI.pth'
IMAGE_FOLDER = 'E'
MASK_FOLDER = ''

# ========== 2. Load Configuration File ==========
with open(CONFIG_PATH, "r") as file:
    cfg = yaml.safe_load(file)
print("✔ Configuration file loaded successfully.")

# Create HIFI model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = HIFI(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("✔ Model loaded and set to inference mode.")

# ========== 4. Preprocessing Function ==========
def preprocess(image_path):
    image = cv2.imread(image_path, 1)  # Read image (BGR format)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)  # Resize
    image = image / 255.0  # Normalize to [0,1]
    image = np.moveaxis(image, 2, 0)  # Convert HWC → CHW
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor

    # Normalization
    normalize = transforms.Normalize(mean=cfg["dataset_params"]["mean"], std=cfg["dataset_params"]["std"])
    image = normalize(image)
    return image.unsqueeze(0)  # Add batch dimension

# ========== 5. Read File Lists ==========
image_files = sorted(os.listdir(IMAGE_FOLDER))
mask_files = sorted(os.listdir(MASK_FOLDER))

# Ensure matching file counts
assert len(image_files) == len(mask_files), "❌ Number of images and masks does not match!"

# Storage for evaluation metrics
auc_scores = []
f1_scores = []
iou_scores = []
recall_scores = []
precision_scores = []
dice_scores = []
exact_match_scores = []
mae_scores = []

skip_count = 0  # Count images skipped for AUC

# ========== 6. Iterate Through All Images ==========
for img_file, mask_file in zip(image_files, mask_files):
    image_path = os.path.join(IMAGE_FOLDER, img_file)
    mask_path = os.path.join(MASK_FOLDER, mask_file)

    print(f"🔹 Processing: {img_file} <--> {mask_file}")

    # Preprocess input image
    input_batch = preprocess(image_path).to(device)

    # ========== Read and Binarize Ground Truth Mask ==========
    gt_mask = np.array(Image.open(mask_path).convert("L"))  
    gt_mask = cv2.resize(gt_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_mask > 127).astype(np.float32)
    gt_mask_flat = gt_mask.flatten()

    # ========== Model Inference ==========
    with torch.no_grad():
        output, edge_output, *_ = model(input_batch)

    # ========== Process Model Output ==========
    output = torch.sigmoid(output).squeeze().cpu().numpy()  
    pred_mask = (output > 0.5).astype(np.float32)  
    pred_mask = cv2.resize(pred_mask, (256, 256), interpolation=cv2.INTER_NEAREST)  
    pred_mask = (pred_mask > 0.5).astype(np.float32)  
    pred_mask_flat = pred_mask.flatten()

    # ========== 7. Compute AUC ==========
    try:
        auc_score = roc_auc_score(gt_mask_flat, pred_mask_flat)
        auc_scores.append(auc_score)
    except ValueError:
        print(f"⚠️ Skipping {img_file}, AUC calculation failed (single-class case).")
        skip_count += 1
        auc_score = None

    # ========== 8. Compute F1-score ==========
    f1 = f1_score(gt_mask_flat, pred_mask_flat, zero_division=1)
    f1_scores.append(f1)

    # ========== 9. Compute IoU ==========
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = intersection / union if union > 0 else 0
    iou_scores.append(iou)

    # ========== 10. Compute Dice Coefficient ==========
    dice = (2 * intersection) / (gt_mask.sum() + pred_mask.sum()) if (gt_mask.sum() + pred_mask.sum()) > 0 else 0
    dice_scores.append(dice)

    # ========== 11. Compute Recall and Precision ==========
    recall = recall_score(gt_mask_flat, pred_mask_flat, zero_division=1)
    precision = precision_score(gt_mask_flat, pred_mask_flat, zero_division=1)
    recall_scores.append(recall)
    precision_scores.append(precision)

    # ========== 12. Compute Exact Match ==========
    exact_match = np.mean(gt_mask_flat == pred_mask_flat)
    exact_match_scores.append(exact_match)

    # ========== 13. Compute MAE (based on binary masks) ==========
    mae = np.mean(np.abs(gt_mask - pred_mask))
    mae_scores.append(mae)

    # Metrics summary for this image
    auc_display = f"{auc_score:.4f}" if auc_score is not None else "Skipped"
    print(
        f"✔ {img_file}: AUC = {auc_display}, F1 = {f1:.4f}, IoU = {iou:.4f}, "
        f"Dice = {dice:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}, "
        f"Exact Match = {exact_match:.4f}, MAE = {mae:.4f}"
    )

# ========== 14. Compute Overall Performance ==========
print("\n========== 📊 Overall Model Evaluation ==========")
print(f"✔ Mean AUC = {np.mean(auc_scores) if auc_scores else 0:.4f}")
print(f"✔ Mean F1-score = {np.mean(f1_scores):.4f}")
print(f"✔ Mean IoU = {np.mean(iou_scores):.4f}")
print(f"✔ Mean Dice = {np.mean(dice_scores):.4f}")
print(f"✔ Mean Recall = {np.mean(recall_scores):.4f}")
print(f"✔ Mean Precision = {np.mean(precision_scores):.4f}")
print(f"✔ Mean Exact Match = {np.mean(exact_match_scores):.4f}")
print(f"✔ Mean MAE = {np.mean(mae_scores):.4f}")
print(f"⚠ {skip_count} images skipped during AUC calculation (single-class issue).")
