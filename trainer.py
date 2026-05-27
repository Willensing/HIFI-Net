import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import utils.utils as utils
from sklearn import metrics
from model.model import HIFI
from utils.utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
from torch.optim.lr_scheduler import StepLR
import os
import datetime
import yaml
from torch import nn

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set random seed
set_random_seed(1221)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load configuration file
config_path ='config\config.yaml'
with open(config_path, 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

# Log file generation
now = datetime.datetime.now()
filename_log = f'Results-{now}.txt'

# Data loader
from dataloader.myLoader import generator

gnr = generator(cfg)
training_generator = gnr.get_train_generator()
validation_generator = gnr.get_val_generator()

# Initialize model
model = HIFI(device).to(device)

# Optimizer settings
if cfg['model_params']['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=cfg['model_params']['lr'], weight_decay=1e-4, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=cfg['model_params']['lr'])

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

# Loss function
criterion = nn.BCEWithLogitsLoss()

def mcrs_loss(
    tar,         
    proj,          
    hm,              
    cfg,
    device=None
):
    cp = cfg['contrastive_params']
    p_len = cp['p_len']
    temperature = cp['temperature']
    con_alpha = cfg['model_params']['con_alpha']
    in_out_weight = cp['in_out_weight']
    reliable_thresh = cp['reliable_thresh']
    gamma = cp['gamma']

    # Patch features
    feat = F.avg_pool2d(proj, kernel_size=p_len, stride=p_len) 
    B, C, h, w = feat.shape
    feat = feat.view(B, C, -1).transpose(1, 2)                 
    feat = F.normalize(feat, dim=-1)

    # Patch soft label and mask
    tar = F.avg_pool2d(tar, kernel_size=p_len, stride=p_len)    
    hm = F.avg_pool2d(hm, kernel_size=p_len, stride=p_len)    
    tar = tar.view(B, 1, -1)                                     
    hm = hm.view(B, 1, -1)                                      

    # Contrastive loss
    c_loss = utils.MCRS(
        feat=feat,
        mask=tar,
        valid_mask=hm,
        temperature=temperature,
        in_out_weight=in_out_weight,
        reliable_thresh=reliable_thresh,
        gamma=gamma,
    )

    return con_alpha * c_loss


# Save best-performing models
max_val_auc = 0
max_val_iou = 0.0  # Initialize as a scalar

best_model_dir = 'bestModels'
os.makedirs(best_model_dir, exist_ok=True)

# Training process
if __name__ == '__main__':
    for epoch in range(cfg['model_params']['epoch']):
        train_inter = AverageMeter()
        train_union = AverageMeter()
        train_sloss = AverageMeter()
        print(f"Training epoch {epoch+1} started")

        # Training loop
        for sample in tqdm(training_generator):
            model.train()
            optimizer.zero_grad()

            img = sample[0].to(device)
            tar = sample[1].to(device).float().unsqueeze(1)
            edge_tar = sample[2].to(device).float().unsqueeze(1)

            pred, edge_out,  mask1, mask2, mask3, feat1, feat2, feat3, hm1, hm2, hm3 = model(img)

            pred = F.interpolate(pred, img.shape[2:], mode='bilinear', align_corners=True)
            edge_out = F.interpolate(edge_out, img.shape[2:], mode='bilinear', align_corners=True)

            mask1 = F.interpolate(mask1, img.shape[2:], mode='bilinear', align_corners=True)
            mask2 = F.interpolate(mask2, img.shape[2:], mode='bilinear', align_corners=True)
            mask3 = F.interpolate(mask3, img.shape[2:], mode='bilinear', align_corners=True)
            feat1 = F.interpolate(feat1, img.shape[2:], mode='bilinear', align_corners=True)
            feat2 = F.interpolate(feat2, img.shape[2:], mode='bilinear', align_corners=True)
            feat3 = F.interpolate(feat3, img.shape[2:], mode='bilinear', align_corners=True)
            hm1 = F.interpolate(hm1, img.shape[2:], mode='bilinear', align_corners=True)
            hm2 = F.interpolate(hm2, img.shape[2:], mode='bilinear', align_corners=True)
            hm3 = F.interpolate(hm3, img.shape[2:], mode='bilinear', align_corners=True)

            mloss1 = mcrs_loss(tar, feat1, hm1, cfg, device=device)
            mloss2 = mcrs_loss(tar, feat2, hm2, cfg, device=device)
            mloss3 = mcrs_loss(tar, feat3, hm3, cfg, device=device)

            # Compute losses
            seg_loss = criterion(pred, tar)
            edge_loss = F.binary_cross_entropy_with_logits(edge_out, edge_tar)

            seg_loss1 = criterion(mask1, tar)
            seg_loss2 = criterion(mask2, tar)
            seg_loss3 = criterion(mask3, tar)
            total_loss = seg_loss + edge_loss  + seg_loss1 + seg_loss2 + seg_loss3 + mloss1 + mloss2 + mloss3

            total_loss.backward()
            optimizer.step()

            train_sloss.update(total_loss.item())
            intr, uni = batch_intersection_union(pred, tar)
            train_inter.update(intr)
            train_union.update(uni)

        scheduler.step()
        train_softmax = train_sloss.avg
        train_IoU = np.mean((train_inter.sum / (train_union.sum + 1e-10)).tolist())

        print(f"Validation epoch {epoch+1} started")

        with torch.no_grad():
            model.eval()
            val_inter = AverageMeter()
            val_union = AverageMeter()
            auc = []

            for img, tar, edge_tar in tqdm(validation_generator):
                img, tar, edge_tar = img.to(device), tar.to(device).float().unsqueeze(1), edge_tar.to(device).float().unsqueeze(1)

                pred, edge_out, _, _, _, _, _, _, _, _, _ = model(img)
                pred = F.interpolate(pred, img.shape[2:], mode='bilinear', align_corners=True)
                edge_out = F.interpolate(edge_out, img.shape[2:], mode='bilinear', align_corners=True)

                pred = (pred > 0.5).float()
                tar = (tar > 0.5).float()
                edge_out = (edge_out > 0.5).float()
                edge_tar = (edge_tar > 0.5).float()

                intr, uni = batch_intersection_union(pred, tar)
                val_inter.update(intr)
                val_union.update(uni)

                y_score = torch.sigmoid(pred)
                for yy_true, yy_pred in zip(tar.cpu().numpy(), y_score.cpu().numpy()):
                    this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
                    that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1 - yy_pred).ravel())
                    auc.append(max(this, that))

            val_auc = np.mean(auc)

            # Save the model with the best AUC
            if val_auc > max_val_auc:
                max_val_auc = val_auc
                best_auc_model_path = os.path.join(best_model_dir, f'model_best_auc_epoch{epoch}.pth')
                torch.save(model.state_dict(), best_auc_model_path)

            # Compute IoU
            val_IoU = np.mean((val_inter.sum / (val_union.sum + 1e-10)).tolist())

            # Save the model with the best IoU
            if val_IoU > max_val_iou:
                max_val_iou = val_IoU
                best_iou_model_path = os.path.join(best_model_dir, f'model_best_iou_epoch{epoch}.pth')
                torch.save(model.state_dict(), best_iou_model_path)

            # Write logs
            logs = {
                'epoch': epoch,
                'Softmax Loss': train_softmax,
                'Train IoU': train_IoU,
                'Validation IoU': val_IoU,
                'Validation AUC': val_auc,
                'Max Validation AUC': max_val_auc,
                "Max IoU Tampered": max_val_iou
            }
            write_logger(filename_log, cfg, **logs)
