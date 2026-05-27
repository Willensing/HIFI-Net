import os
import torch.nn.functional as F
import numpy as np
import torch
import random

def batch_intersection_union(predict, target):
    # 二值化：确保预测和目标都是0或1
    predict = (predict > 0.5).long()  # 将预测值二值化
    target = (target > 0.5).long()    # 将目标标签二值化

    # 计算交集和并集
    intersection = (predict * target).sum()  # 交集部分（预测和真实标签都为1的区域）
    union = (predict + target).sum()        # 并集部分（预测或真实标签为1的区域）

    # 返回交集和并集
    return intersection, union


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count
            
def write_logger(filename_log, cfg, **kwargs):
    if not os.path.isdir('home/user1/HIFI-Net/results'):
        os.mkdir('home/user1/HIFI-Net/results')
    f = open('home/user1/HIFI-Net/results/' + filename_log, "a")
    if kwargs['epoch'] == 0:
        f.write("Training CONFIGS: (Model fixed to ResNet50)\n\n")

    for key, value in kwargs.items():
        f.write(str(key) + ": " + str(value) + "\n")
    f.write("\n")
    f.close()

def set_random_seed(seed, deterministic=False):
    """Set random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def MCRS(
    feat,              # [B, N, D]
    mask,              # [B, 1, N] soft label
    valid_mask,        # [B, 1, N] high-suspicion region mask
    temperature=0.6,
    in_out_weight=0.3,
    reliable_thresh=0.05,
    gamma=2.0,        
):
    B, N, D = feat.shape
    device = feat.device

    # 相似度矩阵计算
    sim = torch.bmm(feat, feat.transpose(1, 2)) / temperature
    contrast_prob = F.softmax(sim, dim=-1)

    # squeeze to [B, N]
    mask = mask.squeeze(1)
    valid_mask = valid_mask.squeeze(1)

    # 1. 区域关系掩码
    outer_mask = 1.0 - valid_mask
    mask_in_in = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
    mask_in_out = valid_mask.unsqueeze(2) * outer_mask.unsqueeze(1) + \
                  outer_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
    region_pair_mask = mask_in_in + in_out_weight * mask_in_out

    # 2. 可信 label 掩码
    reliable_mask = ((mask < reliable_thresh) | (mask > 1 - reliable_thresh)).float()
    r_i = reliable_mask.unsqueeze(2)
    r_j = reliable_mask.unsqueeze(1)
    reliable_pair_mask = r_i * r_j

    # 3. 自身屏蔽
    identity = torch.eye(N, device=device).unsqueeze(0)
    non_diag_mask = 1.0 - identity

    # 4. soft label 生成 soft pair 权重（加上 gamma 幂）
    m_i = mask.unsqueeze(2)
    m_j = mask.unsqueeze(1)
    soft_weight = (1.0 - torch.abs(m_i - m_j)) ** gamma

    # 5. 最终权重
    final_weight = soft_weight * region_pair_mask * reliable_pair_mask * non_diag_mask

    # 6. contrastive loss
    loss_matrix = -torch.log(contrast_prob + 1e-8) * final_weight
    loss = loss_matrix.sum(dim=-1) / (final_weight.sum(dim=-1) + 1e-8)

    return loss.mean()