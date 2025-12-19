import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, in_samples, out_dim, F1=8, D=2, kernel_length=64, dropout_prob=0.55):
        super().__init__()
        F2 = F1 * D
        self.in_samples = in_samples 

        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(dropout_prob)

        self.depthwise_conv2 = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding=(0, 8), bias=False)
        self.pointwise_conv2 = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout_prob)

        self.dense_block = DenseBlock2D(F2, num_layers=3, growth_rate=16)
        self.transition_conv = nn.Conv2d(F2 + 3 * 16, F2, kernel_size=1)

        self.pool2 = None 
        self.final_proj = None  
        self.out_dim = out_dim
        self.F2 = F2
        self.se1 = SEBlock(F2)

    def forward(self, x): 
        B, C, T = x.shape
        x = x.unsqueeze(1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.se1(x) 
        x = self.dense_block(x)
        x = self.transition_conv(x)

        time_len = x.shape[-1]
        pool2_kernel = min(8, time_len)
        if self.pool2 is None or self.pool2.kernel_size[1] != pool2_kernel:
            self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2_kernel)).to(x.device)

        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(B, -1) 

        if self.final_proj is None:
            in_feat = x.shape[1]
            self.final_proj = nn.Linear(in_feat, self.out_dim).to(x.device)

        x = self.final_proj(x)  
        return x


class classifier(nn.Module):
    def __init__(self,out_dim):
        super(classifier, self).__init__()
        self.lead_branches = nn.ModuleList([
            self._make_cnn_branch(in_channels=23),   
            self._make_cnn_branch(in_channels=7),   
            self._make_cnn_branch(in_channels=15),  
            self._make_cnn_branch(in_channels=9),   
            self._make_cnn_branch(in_channels=10),  
        ])

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.out_dim = out_dim
        self.fc = nn.Linear(5 * 32, self.out_dim)

    def _make_cnn_branch(self, in_channels):
        layers = []
        layers.append(nn.Conv1d(in_channels, 32, kernel_size=100, stride=2))
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=1))

        layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=22, stride=1, groups=32))
        layers.append(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=1, stride=1))
        layers.append(nn.BatchNorm1d(32))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=1))

        # Dense Block 1
        layers.append(DenseBlock(32, num_layers=2, growth_rate=16)) 
        layers.append(nn.Conv1d(32 + 2 * 16, 32, kernel_size=1, stride=1)) 


        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Dense Block 2
        dense2 = DenseBlock(32, num_layers=7, growth_rate=16)
        layers.append(ResidualBlockWithConv(
            block=dense2,
            in_channels=32,
            out_channels=32 + 7 * 16,
            projection=True
        ))
        layers.append(nn.Conv1d(32 + 7 * 16, 144, kernel_size=3, stride=1, groups=144))
        layers.append(nn.Conv1d(144, 16, kernel_size=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(kernel_size=2, stride=4))

        # Dense Block 3
        layers.append(DenseBlock(16, num_layers=2, growth_rate=8))  
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)


    def forward(self, x):
        lead_indices_list = [
            [0, 1, 2, 3, 5, 6, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 56, 58, 59, 60, 61, 62],          
            [7, 22, 23, 38, 39, 54, 55],                                                            
            [10, 11, 12, 13, 17, 18, 20, 21, 41, 42, 43, 49, 50, 51, 52],                                        
            [4, 14, 15, 16, 44, 45, 46, 47, 48],                                                                                
            [8, 9, 19, 24, 25, 36, 40, 53, 57,63]  ,                                                      
        ]

        branch_outputs = []
        for idx, (indices, branch) in enumerate(zip(lead_indices_list, self.lead_branches)):
            input_x = x[:, indices, :]
            out = branch(input_x) 
            out = self.global_avg_pool(out) 
            branch_outputs.append(out)

        concat_output = torch.cat(branch_outputs, dim=1)


        x = self.dropout(concat_output)    
        x = self.flatten(x)                    

        x = self.fc(x)                               
        return x

class ResidualBlockWithConv(nn.Module):
    def __init__(self, block, in_channels, out_channels, projection=False):
        super(ResidualBlockWithConv, self).__init__()
        self.block = block
        self.projection = projection

        if projection or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        return out + identity


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = self._make_layer(in_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def _make_layer(self, in_channels, growth_rate):
        return DepthwiseSeparableConv1d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DenseBlock2D(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock2D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DepthwiseSeparableConv2d(
                in_channels + i * growth_rate, growth_rate,
                kernel_size=3, stride=1, padding=1
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class PSDModule(nn.Module):
    def __init__(self, in_channels=64, in_samples=8, out_dim=4):
        super().__init__()
        self.net = FeatureExtractor(in_channels, in_samples, out_dim)

    def forward(self, x):  # (B, C, F1)
        return self.net(x)

class WaveletModule(nn.Module):
    def __init__(self, in_channels=64, in_samples=5, out_dim=4):
        super().__init__()
        self.net = FeatureExtractor(in_channels, in_samples, out_dim)

    def forward(self, x):  # (B, C, F2)
        return self.net(x)

class NonlinearModule(nn.Module):
    def __init__(self, in_channels=64, in_samples=6, out_dim=4):
        super().__init__()
        self.net = FeatureExtractor(in_channels, in_samples, out_dim)

    def forward(self, x):  # (B, C, F3)
        return self.net(x)
    
class CoherenceModule(nn.Module):
    def __init__(self, in_channels=64, in_samples=64, out_dim=4):
        super().__init__()
        self.net = FeatureExtractor(in_channels, in_samples, out_dim)

    def forward(self, x):  # (B, 19, 19)
        return self.net(x)
    
class RawEEGModule(nn.Module):
    def __init__(self,out_dim=4):
        super().__init__()
        self.net = classifier(out_dim)

    def forward(self, x):  # (B, 19, 2000)
        return self.net(x)

class GMMFusion(nn.Module):
    def __init__(self, in_dim=4, num_components=4):
        super().__init__()
        self.num_components = num_components
        self.means = nn.Parameter(torch.randn(num_components, in_dim))
        self.log_vars = nn.Parameter(torch.zeros(num_components, in_dim))

    def forward(self, features):
        # features: (B, M, D)

        B, M, D = features.size()

        feats_exp = features.unsqueeze(2)             # (B, M, 1, D)
        means = self.means.unsqueeze(0).unsqueeze(0)  # (1, 1, K, D)
        vars_ = torch.exp(self.log_vars).unsqueeze(0).unsqueeze(0) + 1e-6
        sq = (feats_exp - means)**2
        exp_term = torch.exp(-0.5 * sq / vars_)       # (B, M, K, D)
        gauss = torch.prod(exp_term, dim=3)           # (B, M, K)
        resp = gauss / (gauss.sum(dim=2, keepdim=True) + 1e-6)  # (B, M, K)
        new_feats = (resp.unsqueeze(3) * means).sum(dim=2)     # (B, M, D)

        fused = new_feats.mean(dim=1)             # (B, D)
        orig_mean = features.mean(dim=1)          # (B, D)
        mse = F.mse_loss(fused, orig_mean, reduction='mean')  # scalar
        return new_feats, mse
    

class ScoreEvaluation(nn.Module):
    def __init__(self, in_features=4, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, recon_feats):
        B, M, D = recon_feats.size()
        flat = recon_feats.view(B*M, D)
        logits = self.classifier(flat)           # (B*M, C)
        probs = F.softmax(logits, dim=1)
        logits = logits.view(B, M, -1)           # (B, M, C)
        probs = probs.view(B, M, -1)
        return logits, probs

def mutual_information_fusion_logits(logits_matrix, eps=1e-6):
    prob_matrix = F.softmax(logits_matrix, dim=-1)    
    ent = -torch.sum(prob_matrix * torch.log(prob_matrix + eps), dim=-1)  # (B, M)
    w = 1.0 / (ent + eps)                             
    w = w / w.sum(dim=1, keepdim=True)                 # (B, M)
    fused_logits = torch.sum(w.unsqueeze(-1) * logits_matrix, dim=1)  # (B, C)
    return fused_logits


class HACENet(nn.Module):
    def __init__(self, expert_out_dim=4, num_classes=2, use_fusion=True, configs=None):
        super().__init__()
        self.expert_psd       = PSDModule(in_samples=8, out_dim=expert_out_dim)
        self.expert_wavelet   = WaveletModule(in_samples=5, out_dim=expert_out_dim)
        self.expert_nonlinear = NonlinearModule(in_samples=6, out_dim=expert_out_dim)
        self.expert_raw       = RawEEGModule(out_dim=expert_out_dim)
        self.expert_coherence = CoherenceModule(in_samples=64, out_dim=expert_out_dim)

        self.use_fusion = use_fusion

        self.fusion_psd       = GMMFusion(in_dim=expert_out_dim, num_components=4)
        self.fusion_wavelet   = GMMFusion(in_dim=expert_out_dim, num_components=4)
        self.fusion_nonlinear = GMMFusion(in_dim=expert_out_dim, num_components=4)
        self.fusion_raw       = GMMFusion(in_dim=expert_out_dim, num_components=4)
        self.fusion_coherence = GMMFusion(in_dim=expert_out_dim, num_components=4)

        self.score_eval = ScoreEvaluation(in_features=expert_out_dim, num_classes=num_classes)

    def forward(self, psd_feats, wavelet_feats, nonlinear_feats, raw_eeg, coherence_feats):
        f1 = self.expert_psd(psd_feats)
        f2 = self.expert_wavelet(wavelet_feats)
        f3 = self.expert_nonlinear(nonlinear_feats)
        f4 = self.expert_raw(raw_eeg)
        f5 = self.expert_coherence(coherence_feats)


        r1, l1 = self.fusion_psd(f1.unsqueeze(1))
        r2, l2 = self.fusion_wavelet(f2.unsqueeze(1))
        r3, l3 = self.fusion_nonlinear(f3.unsqueeze(1))
        r4, l4 = self.fusion_raw(f4.unsqueeze(1))
        r5, l5 = self.fusion_coherence(f5.unsqueeze(1))

        recon_feats = torch.cat([r1, r2, r3, r4, r5], dim=1)
        recon_loss = l1 + l2 + l3 + l4 + l5

        logits_mat, _ = self.score_eval(recon_feats)

        if self.use_fusion:
            fused_logits = mutual_information_fusion_logits(logits_mat)
        else:
            fused_logits = logits_mat.mean(dim=1)

        return logits_mat, fused_logits, recon_loss



