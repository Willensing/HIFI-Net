import torch
import torch.nn as nn
import torch.nn.functional as F
from .aspp import build_aspp
from .srm import setup_srm_layer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_U2net import U2NET
import torchvision.models as models
from scipy.ndimage import distance_transform_edt as distance_transform

# Basic convolution block
class ConvB(nn.Module):
    def __init__(self, in_channels, out_channels, size, stride, padding=1):
        super(ConvB, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SkipBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


# Bottleneck module
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.rate = rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# ResNet50 module
class ResNet(nn.Module):
    def __init__(self, n_classes=1000, os=16, pretrained=True):
        super(ResNet, self).__init__()

        self.inplanes = 64
        
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 4]

        # Load pretrained ResNet50
        resnet50 = models.resnet50(pretrained=pretrained)

        # Copy layers from the pretrained model into this model
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        # Use custom make_layer while keeping original architecture
        self.layer1 = self._make_layer(64, 3, stride=strides[0], rate=rates[0])
        self.layer2 = self._make_layer(128, 4, stride=strides[1], rate=rates[1])
        self.layer3 = self._make_layer(256, 6, stride=strides[2], rate=rates[2])
        self.layer4 = self._make_layer(512, 3, stride=strides[3], rate=rates[3])

    def _make_layer(self, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x1 = self.relu(x)
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1, x2, x3, x4, x5

class FRW(nn.Module):
    def __init__(self, in_channels):
        super(FRW, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Compute channel attention
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))  # Global average pooling
        max_pool = torch.max(x, dim=2, keepdim=True)[0]  # Global max pooling
        max_pool = torch.max(max_pool, dim=3, keepdim=True)[0]  # Global max pooling

        avg_out = self.conv2(self.relu(self.conv1(avg_pool)))
        max_out = self.conv2(self.relu(self.conv1(max_pool)))
        
        # Channel weighting
        rw = avg_out + max_out
        rw = self.sigmoid(rw)
        x_fused = x * rw
        return x_fused

class ContextualAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(ContextualAttention, self).__init__()
        mid_channels = in_channels // reduction

        # QKV encoding
        self.query_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        # Output fusion
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).view(B, -1, H * W).transpose(1, 2)  # (B, H*W, C//r)
        proj_key = self.key_conv(x).view(B, -1, H * W)                      # (B, C//r, H*W)
        proj_value = self.value_conv(x).view(B, C, H * W)                   # (B, C, H*W)

        attention = torch.bmm(proj_query, proj_key)                         # (B, H*W, H*W)
        attention = self.softmax(attention)

        out = torch.bmm(proj_value, attention.transpose(1, 2))             # (B, C, H*W)
        out = out.view(B, C, H, W)
        out = self.out_proj(out)

        # Residual enhanced features
        return out + x

class NRJC(nn.Module):
    def __init__(self, in1, in2, k1, s1):
        super(NRJC, self).__init__()
        self.conv1 = ConvB(in1, in2, k1, s1)  # Process low-level feature x1
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # Upsampling
        self.conv2 = ConvB(in2, in2, 3, 1)  # Process encoder feature x2
        self.conv3 = ConvB(in2 * 2, in2, 3, 1)
        self.frw = FRW(64)
        self.CATT1 = ContextualAttention(64)
        self.CATT2 = ContextualAttention(64)

    def forward(self, x1, x2, srm):
        x1 = self.conv1(x1)  # Process low-level features
        x1 = self.upsample(x1)  # Upsample to match x2 size
        srm = self.frw(srm)
        x2 = srm + x2
        x2 = self.conv2(x2)  # Process encoder features
        x1 = self.CATT1(x1)
        x2 = self.CATT2(x2)
        x = torch.cat([x1, x2], dim=1)  # U-Net style skip connection
        x = self.conv3(x)  # 3x3 convolution to fuse features
        return x  # Return fused features

class SIM(nn.Module):
    def __init__(self, init_edge_margin, init_weights=[1.0, 1.3, 1.9]):
        super(SIM, self).__init__()
        self.region_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        self.edge_margin = float(init_edge_margin)

    def forward(self, X):
        """
        Input:
            X: Tensor of shape [B, 1, H, W], arbitrary value range (e.g. logits)
        Output:
            weight_map: Tensor [B, 1, H, W], per-pixel region weight
            suspicious_mask: Tensor [B, 1, H, W], high-value region mask (X_norm >= 0.5)
        """
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input X must be a torch.Tensor")

        B, C, H, W = X.shape
        if C != 1:
            raise ValueError("Input X must have shape [B, 1, H, W]")

        # Normalization
        X_min = X.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        X_max = X.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        X_norm = (X - X_min) / (X_max - X_min + 1e-8)

        # Binarization
        binary_mask = (X_norm > 0.5).float()

        # Sobel edge extraction
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32, device=X.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32, device=X.device).unsqueeze(0).unsqueeze(0)

        grad_x = F.conv2d(binary_mask, sobel_x, padding=1)
        grad_y = F.conv2d(binary_mask, sobel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        edge_map = (grad_magnitude > 0.1).float()

        # Distance transform
        dist_map = torch.zeros_like(X_norm)
        for i in range(B):
            edge_np = edge_map[i, 0].cpu().numpy().astype(bool)
            dist = distance_transform(~edge_np)
            dist_map[i, 0] = torch.from_numpy(dist).to(X.device)

        # Region classification
        region_map = torch.zeros_like(X_norm, dtype=torch.long)

        edge_margin_val = max(self.edge_margin, 0.0)
        region_map[dist_map <= edge_margin_val] = 1

        high_mask = (dist_map > edge_margin_val) & (X_norm >= 0.5)
        low_mask = (dist_map > edge_margin_val) & (X_norm < 0.5)

        region_map[high_mask] = 2
        region_map[low_mask] = 0

        weight_map = self.region_weights[region_map]
        suspicious_mask = (X_norm >= 0.5).float()

        return weight_map, suspicious_mask

class SGFM(nn.Module):
    def __init__(self, edgemargin, inchannels=64):
        super(SGFM, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(inchannels, inchannels, 1)
        self.frw = FRW(64)
        self.convS = nn.Sequential(
                        nn.Conv2d(64, 1, kernel_size=7, padding=3),
                        nn.Sigmoid())
                    
        self.conv2 = nn.Conv2d(inchannels, 1, kernel_size=3, stride=1, padding=1)
        self.sim = SIM(edgemargin)

    def forward(self, x, clue):
        x2 = self.conv1(x)
        clue = x2 * clue
        x = x + clue
        x = self.up(x)
        x = self.frw(x)
        weight = self.convS(x)
        x = x * weight
        out = self.conv2(x)
        mask = self.up(out)
        weight_map, suspicious_mask = self.sim(out)

        return mask, weight_map, x, suspicious_mask

class Edge(nn.Module):
    def __init__(self):
        super(Edge, self).__init__()
        self.conv1 = ConvB(64, 64, 3, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        edge_feat = self.up(x)
        x = self.conv3(edge_feat)  # Final edge prediction
        return edge_feat, x  # Return enhanced edge feature and edge detection result
    
class Body(nn.Module):
    def __init__(self):
        super(Body, self).__init__()
        self.conv1 = ConvB(64, 64, 3, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x, et):
        x = self.conv1(x)
        x = self.up(x)
        x = x + et
        x = self.conv3(x)  # Final prediction map
        return x

# U2NET model (load and freeze all parameters)
class U2NETWrapper(nn.Module):
    def __init__(self, device):
        super(U2NETWrapper, self).__init__()
        model_path='model_U2net\\u2net.pth'
        self.device = device  
        self.model = U2NET(3, 1)  
        

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)  
        self.model.eval()  

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
       
        x = x.to(self.device)
        with torch.no_grad():
            output = self.model(x)  
            pred = output[0] 
            return pred

# Tampering detection network
class HIFI(nn.Module):
    def __init__(self, device):
        super(HIFI, self).__init__()

        self.u2net = U2NETWrapper(device=device)
        
        self.conv_srm = setup_srm_layer()
        self.encoder1 = ResNet() 
        self.encoder2 = ResNet()


        
        self.skip1 = SkipBlock(2048, 256)
        self.skip_s = SkipBlock(2048, 256)
        self.skip2 = SkipBlock(1024, 64)
        self.skip2s = SkipBlock(1024, 64)
        self.skip3 = SkipBlock(512, 64)
        self.skip3s = SkipBlock(512, 64)
        self.skip4 = SkipBlock(256, 64)
        self.skip4s = SkipBlock(256, 64)



        self.aspp = build_aspp(inplanes=512, outplanes=64)
        self.nrjc3 = NRJC(64, 64, 4, 2)
        self.nrjc2 = NRJC(64, 64, 3, 1)
        self.nrjc1 = NRJC(64, 64, 3, 1)

        self.premap = SIM(5)

        self.avg_pool = nn.AvgPool2d(kernel_size=16, stride=16)
        self.sgfm3 = SGFM(2)
        self.sgfm2 = SGFM(4)
        self.sgfm1 = SGFM(6)

        self.edge1 = Edge()
        self.edge2 = Edge()
        self.edge3 = Edge()
        self.body1 = Body()
        self.body2 = Body()
        self.body3 = Body()

        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Learnable fusion weights
        self.em_weights = nn.Parameter(torch.tensor([7.0, 5.0, 2.0]), requires_grad=True)
        self.bm_weights = nn.Parameter(torch.tensor([7.0, 5.0, 2.0]), requires_grad=True)

    def forward(self, x):

        Saliency_Prior = self.u2net(x)

        x_srm = self.conv_srm(x)

        _, srm2, srm3, srm4, srm5 = self.encoder1(x_srm)
        _, x2, x3, x4, x5 = self.encoder2(x)


        x5 = self.skip1(x5)
        srm5 = self.skip_s(srm5)
        x4 = self.skip2(x4)
        x3 = self.skip3(x3)
        x2 = self.skip4(x2)
        srm4 = self.skip2s(srm4)
        srm3 = self.skip3s(srm3)
        srm2 = self.skip4s(srm2)

        x = torch.concat((x5, srm5), dim=1)
        x = self.aspp(x)

        map4, _ = self.premap(Saliency_Prior)
        x = self.nrjc3(x, x4, srm4)
        '''
        return mask, weight_map, x, suspicious_mask
        mask, MAP, x, proj, highmask
        mask1, map1, t1, proj1, hm1
        t1 = proj1
        '''
        map4 = self.avg_pool(map4)
        mask3, weight_map3, t3, suspicious_mask3 = self.sgfm3(x, map4)
        # mask1, map1, t1, proj1, hm1= self.sawm1(s1, map0)
        mask3 = self.up3(mask3)
        t3 = self.up3(t3)
        suspicious_mask3 = self.up3(suspicious_mask3)

        x = self.nrjc2(x, x3, srm3)
        # s2 = x.clone()
        mask2, weight_map2, t2, suspicious_mask2 = self.sgfm2(x, weight_map3)
        mask2 = self.up2(mask2)
        t2 = self.up2(t2)
        suspicious_mask2 = self.up2(suspicious_mask2)

        x = self.nrjc1(x, x2, srm2)
        # s3 = x.clone()
        mask1, _, t1, suspicious_mask1 = self.sgfm1(x, weight_map2)

        # Obtain edge and body decoding results
        et3, em3 = self.edge3(t3)
        bm3 = self.body3(t3, et3)


        et2, em2 = self.edge2(t2)
        bm2 = self.body2(t2, et2)

        et1, em1 = self.edge1(t1)
        bm1 = self.body1(t1, et1)

        # Normalize weights
        em_weights = torch.softmax(self.em_weights, dim=0)
        bm_weights = torch.softmax(self.bm_weights, dim=0)

        # Fuse results using learned weights
        em = em_weights[0] * em1 + em_weights[1] * em2 + em_weights[2] * em3
        bm = bm_weights[0] * bm1 + bm_weights[1] * bm2 + bm_weights[2] * bm3

        return bm, em, mask3, mask2, mask1, t3, t2, t1, suspicious_mask3, suspicious_mask2, suspicious_mask1
