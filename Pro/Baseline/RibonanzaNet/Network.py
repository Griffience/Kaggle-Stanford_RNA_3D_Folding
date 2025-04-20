import math
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

import torch.utils.checkpoint as checkpoint


from dropout import *

def init_weights(m):
    #print(m)
    if m is not None and isinstance(m, nn.Linear):
        pass
        # torch.nn.init.xavier_uniform_(m.weight)
        # #torch.nn.init.xavier_normal(m.bias)
        # try:
        #     m.bias.data.fill_(0.01)
        # except:
        #     pass
#mish activation
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'



'''
添加头
'''
class SeparableConv(nn.Module):
    """
    基于深度可分离卷积思想的模块：
    1. depthwise 卷积(groups=in_channels)
    2. pointwise 卷积 (1x1)
    同时结合 LayerNorm 和 Dropout 稳定训练。
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.1):
        super(SeparableConv, self).__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, dilation=dilation, 
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入 x: (B, L, C) —— 转为 (B, C, L) 进行卷积计算
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = x.transpose(1, 2)  # 恢复至 (B, L, C)
        x = self.norm(x)
        x = self.dropout(x)
        return x

class MambaBlock(nn.Module):
    """
    简单的 Mamba 模块，类似于 Transformer 中的前馈网络：
    LayerNorm → Linear → GELU → Dropout → Linear → Dropout,带残差连接。
    需要进一步扩展或引入 gating 机制。
    """
    def __init__(self, channels, hidden_ratio=4, dropout=0.1):
        super(MambaBlock, self).__init__()
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, channels * hidden_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * hidden_ratio, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x

        
'''修改简单Mamba模块'''
class TrueMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ssm = SelectiveSSM(d_model)  # 实现网页10的选择性SSM
        self.conv = CausalConv1D(d_model)  # 因果卷积
        self.gate = nn.Sequential(   #后面仅使用sigmoid激活，没有非线性变换，这里多层非线性提升门控表达能力
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, 2*d_model)
        )
        # self.gate = nn.Linear(d_model, 2*d_model) #动态门控
        
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        conv_out = self.conv(x)
        ssm_out = self.ssm(x)
        return gate[:,0]*conv_out + gate[:,1]*ssm_out

class SelectiveSSM(nn.Module):
    """支持RNA序列建模的选择性SSM实现"""
    def __init__(self, d_model, n_states=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.n_states = n_states
        self.expand = expand
        
        # 输入相关参数生成
        self.proj = nn.Sequential(
            nn.Linear(d_model, 3*d_model + n_states),
            nn.SiLU()
        )
        
        # 结构化矩阵A
        self.A = nn.Parameter(torch.randn(n_states))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # 离散化相关参数
        self.dt_proj = nn.Linear(d_model, 1)
        
    def discretize(self, delta, A, B):
        """离散化过程"""
        # 零阶保持离散化 ZOH
        exp_deltaA = torch.exp(delta.unsqueeze(-1) * A)  #[B,L,N]
        bar_A = exp_deltaA
        bar_B = (exp_deltaA - 1) / (A + 1e-6) * B  #避免除零
        return bar_A, bar_B

    def forward(self, x):
        B, L, _ = x.shape
        
        # 生成动态参数
        params = self.proj(x)  # [B,L,3D+N]
        delta, B, C, A_scale = torch.split(params, 
            [1, self.d_model, self.d_model, self.n_states], dim=-1)
        
        # 参数变换 
        delta = F.softplus(self.dt_proj(delta))  # [B,L,1]
        A = self.A * A_scale  # 输入依赖的A矩阵
        
        # 离散化ZOH
        bar_A, bar_B = self.discretize(delta, A, B)
        
        # 硬件感知扫描  并行递归
        h = torch.zeros(B, self.n_states, device=x.device)
        outputs = []
        for t in range(L):
            # h =( bar_A[:,t] * h + bar_B[:,t] * x[:,t] ).clone()  # [B,N]
            h = (bar_A[:,t].clone() * h.detach() + bar_B[:,t].clone() * x[:,t].clone())
            h = h.clone().requires_grad_()
            y_t = torch.einsum('bn,bn->b', C[:,t], h) + self.D * x[:,t]
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)

class CausalConv1D(nn.Module):
    """因果卷积实现"""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, dilation=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size, 
            padding=0,  # 手动处理padding
            dilation=dilation
        )
    def forward(self, x):
        # 输入格式转换 [B,L,C] -> [B,C,L]
        x = x.permute(0, 2, 1)
        
        # 因果padding（网页7的左侧填充）
        x = F.pad(x, (self.padding, 0))  
        
        # 卷积计算
        x = self.conv(x)
        
        # 恢复维度
        return (x.permute(0, 2, 1)[:, :-self.padding, :] ).clone() # 裁剪尾部padding



def drop_path(x, drop_prob: float = 0., training: bool = False):
    """随机深度丢弃实现"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 保持维度对齐
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 二值化掩码
    output = x.div(keep_prob) * random_tensor  # 数值稳定性处理[4](@ref)
    return output

class DropPath(nn.Module):
    """适用于深度网络的正则化模块"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class SeparableConvMambaBlock(nn.Module):
    """
    多尺度 SeparableConv + Mamba 模块，用于 Early Layers 模块设计
    采用多个不同 kernel_size（或 dilation）的 SeparableConv 模块，
    将多尺度特征拼接后融合，并接入 Mamba 模块增强特征表达。
    """
    def __init__(self, channels, kernel_sizes=[3, 5, 7], dilations=None, dropout=0.1, drop_path_prob=0.1):
        super(SeparableConvMambaBlock, self).__init__()
        if dilations is None:
            dilations = [1 for _ in kernel_sizes]
        self.scales = nn.ModuleList([
            SeparableConv(channels, channels, kernel_size=k, dilation=d, dropout=dropout)
            for k, d in zip(kernel_sizes, dilations)
        ])
        # 使用线性层融合拼接后的多尺度特征，亦可采用 1D 卷积融合
        self.fuse_linear = nn.Linear(len(kernel_sizes) * channels, channels)
        self.mamba = MambaBlock(channels, dropout=dropout)
        self.norm = nn.LayerNorm(channels)
        self.drop_path = DropPath(drop_path_prob) if 'DropPath' in globals() else nn.Identity()

    def forward(self, x):
        # x: (B, L, C)
        outs = [conv(x) for conv in self.scales]
        x_cat = torch.cat(outs, dim=-1)  # 拼接: (B, L, C * num_scales)
        fused = self.fuse_linear(x_cat)
        fused = self.mamba(fused)
        fused = self.norm(fused)
        fused = self.drop_path(fused)
        return fused


'''添加尾'''



class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        #self.gamma=torch.tensor(32.0)

    def forward(self, q, k, v, mask=None, attn_mask=None):

        #print(self.gamma)
        attn = torch.matmul(q, k.transpose(2, 3))/ self.temperature
        #to_plot=attn[0,0].detach().cpu().numpy()
        # plt.imshow(to_plot)
        # plt.show()
        # exit()

        #exit()
        if mask is not None:
            #attn = attn.masked_fill(mask == 0, -1e9)
            #attn = attn#*self.gamma
            # print(attn.shape)
            # print(mask.shape)
            # exit()
            # print(attn.shape)
            # print(mask.shape)
            # exit()
            # print(attn.shape)
            # print(mask.shape)
            # exit()
            
            attn = attn+mask
            # print(attn.shape)
            # exit()


        if attn_mask is not None:
            # print(attn.shape)
            # print(attn_mask.shape)
            # attn = attn+attn_mask
            #attn=attn.float().masked_fill(attn_mask == 0, float('-inf'))
            #pass
            for i in range(len(attn_mask)):
                attn_mask[i,0]=attn_mask[i,0].fill_diagonal_(1)
            # print(attn_mask.shape)
            # exit()
            #print(torch.diagonal(attn_mask).mean())
            attn=attn.float().masked_fill(attn_mask == 0, float('-1e-9'))


        attn = self.dropout(F.softmax(attn, dim=-1))
        # print(attn[0,0])
        # to_plot=attn[0,0].detach().cpu().numpy()
        # with open('mat.txt','w+') as f:
        #     for vector in to_plot:
        #         for num in vector:
        #             f.write('{:04.3f} '.format(num))
        #         f.write('\n')
        # plt.imshow(to_plot)
        # plt.show()
        # exit()
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None,src_mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        if src_mask is not None:
            src_mask=src_mask[:,:q.shape[2]].unsqueeze(-1).float()
            # q=q+src_mask
            # k=k+src_mask
            # print(src_mask.shape)
            # print(src_mask[0])
            attn_mask=torch.matmul(src_mask,src_mask.permute(0,2,1))#.long()
            #attn_mask=attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
            attn_mask=attn_mask.unsqueeze(1)
            # print(attn_mask.shape)
            # exit()
            # print(src_mask.shape)
            #to_plot=attn_mask[1].squeeze().detach().cpu().numpy()
            #plt.imshow(to_plot)
            #plt.show()
            # exit()
            # exit()
            # src_mask
            # src_mask
            #print(q[0,0,:,0])
        #exit()
            q, attn = self.attention(q, k, v, mask=mask,attn_mask=attn_mask)
        else:
            q, attn = self.attention(q, k, v, mask=mask)
        #print(attn.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #print(q.shape)
        #exit()
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class ConvTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, 
                 dim_feedforward, pairwise_dimension, use_triangular_attention, dropout=0.1, k = 3,
                 ):
        super(ConvTransformerEncoderLayer, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        #self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        #self.dropout4 = nn.Dropout(dropout)

        self.pairwise2heads=nn.Linear(pairwise_dimension,nhead,bias=False)
        self.pairwise_norm=nn.LayerNorm(pairwise_dimension)
        self.activation = nn.GELU()

        self.conv=nn.Conv1d(d_model,d_model,k,padding=k//2)

        self.triangle_update_out=TriangleMultiplicativeModule(dim=pairwise_dimension,mix='outgoing')
        self.triangle_update_in=TriangleMultiplicativeModule(dim=pairwise_dimension,mix='ingoing')

        self.pair_dropout_out=DropoutRowwise(dropout)
        self.pair_dropout_in=DropoutRowwise(dropout)


        self.use_triangular_attention=use_triangular_attention
        if self.use_triangular_attention:
            self.triangle_attention_out=TriangleAttention(in_dim=pairwise_dimension,
                                                                    dim=pairwise_dimension//4,
                                                                    wise='row')
            self.triangle_attention_in=TriangleAttention(in_dim=pairwise_dimension,
                                                                    dim=pairwise_dimension//4,
                                                                    wise='col')

            self.pair_attention_dropout_out=DropoutRowwise(dropout)
            self.pair_attention_dropout_in=DropoutColumnwise(dropout)

        self.outer_product_mean=Outer_Product_Mean(in_dim=d_model,pairwise_dim=pairwise_dimension)

        #self.deconv=nn.ConvTranspose1d(d_model,d_model,k)
        self.pair_transition=nn.Sequential(
                                           nn.LayerNorm(pairwise_dimension),
                                           nn.Linear(pairwise_dimension,pairwise_dimension*4),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pairwise_dimension*4,pairwise_dimension))
        
    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(*inputs)
            return inputs
        return custom_forward

    def forward(self, src , pairwise_features, src_mask=None, return_aw=False, use_gradient_checkpoint=False):
        
        src = src*src_mask.float().unsqueeze(-1)

        res = src
        # print(self.norm3(self.conv(src.permute(0,2,1)).permute(0,2,1)).shape)
        # exit()
        src = src + self.conv(src.permute(0,2,1)).permute(0,2,1)
        src = self.norm3(src)
        # print(src.shape)
        # exit()

        pairwise_bias=self.pairwise2heads(self.pairwise_norm(pairwise_features)).permute(0,3,1,2)
        #print(src.shape)
        src2,attention_weights = self.self_attn(src, src, src, mask=pairwise_bias, src_mask=src_mask)
        

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # print(src.shape)
        # exit()
        # print(src_mask)
        # exit()
        if use_gradient_checkpoint:
            pairwise_features=pairwise_features+checkpoint.checkpoint(self.custom(self.outer_product_mean), src)
            pairwise_features=pairwise_features+self.pair_dropout_out(
                checkpoint.checkpoint(self.custom(self.triangle_update_out), pairwise_features, src_mask))
            pairwise_features=pairwise_features+self.pair_dropout_in(
                checkpoint.checkpoint(self.custom(self.triangle_update_in), pairwise_features, src_mask))
            # pairwise_features=pairwise_features+self.pair_dropout_out(self.triangle_update_out(pairwise_features,src_mask))
            # pairwise_features=pairwise_features+self.pair_dropout_in(self.triangle_update_in(pairwise_features,src_mask))
        else:
            pairwise_features=pairwise_features+self.outer_product_mean(src)
            pairwise_features=pairwise_features+self.pair_dropout_out(self.triangle_update_out(pairwise_features,src_mask))
            pairwise_features=pairwise_features+self.pair_dropout_in(self.triangle_update_in(pairwise_features,src_mask))
        if self.use_triangular_attention:
            pairwise_features=pairwise_features+self.pair_attention_dropout_out(self.triangle_attention_out(pairwise_features,src_mask))
            pairwise_features=pairwise_features+self.pair_attention_dropout_in(self.triangle_attention_in(pairwise_features,src_mask))

        if use_gradient_checkpoint:
            pairwise_features=pairwise_features+checkpoint.checkpoint(self.custom(self.pair_transition),pairwise_features)
        else:
            pairwise_features=pairwise_features+self.pair_transition(pairwise_features)
        if return_aw:
            return src,pairwise_features,attention_weights
        else:
            return src,pairwise_features

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Outer_Product_Mean(nn.Module):
    def __init__(self, in_dim=256, dim_msa=32, pairwise_dim=64):
        super(Outer_Product_Mean, self).__init__()
        self.proj_down1 = nn.Linear(in_dim, dim_msa)
        self.proj_down2 = nn.Linear(dim_msa ** 2, pairwise_dim)

    def forward(self,seq_rep, pair_rep=None):
        seq_rep=self.proj_down1(seq_rep)
        outer_product = torch.einsum('bid,bjc -> bijcd', seq_rep, seq_rep)
        outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        outer_product = self.proj_down2(outer_product)

        if pair_rep is not None:
            outer_product=outer_product+pair_rep

        return outer_product 

class relpos(nn.Module):

    def __init__(self, dim=64):
        super(relpos, self).__init__()
        self.linear = nn.Linear(17, dim)

    def forward(self, src):
        L=src.shape[1]
        res_id = torch.arange(L).to(src.device).unsqueeze(0)
        device = res_id.device
        bin_values = torch.arange(-8, 9, device=device)
        #print((bin_values))
        d = res_id[:, :, None] - res_id[:, None, :]
        bdy = torch.tensor(8, device=device)
        d = torch.minimum(torch.maximum(-bdy, d), bdy)
        d_onehot = (d[..., None] == bin_values).float()
        #print(d_onehot.sum(dim=-1).min())
        assert d_onehot.sum(dim=-1).min() == 1
        p = self.linear(d_onehot)
        return p

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, src_mask):
        src_mask=src_mask.unsqueeze(-1).float()
        mask = torch.matmul(src_mask,src_mask.permute(0,2,1))
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


class RibonanzaNet(nn.Module):

    #def __init__(self, ntoken=5, nclass=1, ninp=512, nhead=8, nlayers=9, kmers=9, dropout=0):
    def __init__(self, config):

        super(RibonanzaNet, self).__init__()
        self.config=config
        nhid=config.ninp*4

        self.transformer_encoder = []
        print(f"constructing {config.nlayers} ConvTransformerEncoderLayers")
        for i in range(config.nlayers):
            if i!= config.nlayers-1:
                k=5
            else:
                k=1
            #print(k)
            self.transformer_encoder.append(ConvTransformerEncoderLayer(d_model = config.ninp, nhead = config.nhead,
                                                                        dim_feedforward = nhid, 
                                                                        pairwise_dimension= config.pairwise_dimension,
                                                                        use_triangular_attention=config.use_triangular_attention,
                                                                        dropout = config.dropout, k=k))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(config.ntoken, config.ninp, padding_idx=4)
        self.decoder = nn.Linear(config.ninp,config.nclass)
        # if config.use_bpp:
        #     self.mask_dense=nn.Conv2d(2,config.nhead//4,1)
        # else:
        #     self.mask_dense=nn.Conv2d(1,config.nhead//4,1)

        self.outer_product_mean=Outer_Product_Mean(in_dim=config.ninp,pairwise_dim=config.pairwise_dimension)
        self.pos_encoder=relpos(config.pairwise_dimension)

        self.use_gradient_checkpoint=config.use_grad_checkpoint

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward

    def forward(self, src,src_mask=None,return_aw=False):
        B,L=src.shape
        src = src
        src = self.encoder(src).reshape(B,L,-1)
        
        #spawn outer product
        # outer_product = torch.einsum('bid,bjc -> bijcd', src, src)
        # outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        # print(outer_product.shape)
        if self.use_gradient_checkpoint:
            pairwise_features=checkpoint.checkpoint(self.custom(self.outer_product_mean), src)
            pairwise_features=pairwise_features+self.pos_encoder(src)
        else:
            pairwise_features=self.outer_product_mean(src)
            pairwise_features=pairwise_features+self.pos_encoder(src)
        # print(pairwise_features.shape)
        # exit()

        attention_weights=[]
        for i,layer in enumerate(self.transformer_encoder):
   
                #src_key_padding_mask
            # if return_aw:
            #     src,aw=layer(src, pairwise_features, src_mask,return_aw=return_aw, )
            #     attention_weights.append(aw)
            # else:
            src,pairwise_features=layer(src, pairwise_features, src_mask,return_aw=return_aw,use_gradient_checkpoint=self.use_gradient_checkpoint)

            #print(src.shape)
        output = self.decoder(src).squeeze(-1)+pairwise_features.mean()*0


        if return_aw:
            return output, attention_weights
        else:
            return output
        
    def get_embeddings(self, src,src_mask=None,return_aw=False):
        B,L=src.shape
        src = src
        src = self.encoder(src).reshape(B,L,-1)
        
        #spawn outer product
        # outer_product = torch.einsum('bid,bjc -> bijcd', src, src)
        # outer_product = rearrange(outer_product, 'b i j c d -> b i j (c d)')
        # print(outer_product.shape)
        if self.use_gradient_checkpoint:
            #print("using grad checkpointing")
            pairwise_features=checkpoint.checkpoint(self.custom(self.outer_product_mean), src)
            pairwise_features=pairwise_features+self.pos_encoder(src)
        else:
            pairwise_features=self.outer_product_mean(src)
            pairwise_features=pairwise_features+self.pos_encoder(src)
        # print(pairwise_features.shape)
        # exit()

        attention_weights=[]
        for i,layer in enumerate(self.transformer_encoder):
            # if src_mask is not None:
            #     #src_key_padding_mask
            #     if return_aw:
            #         src,aw=layer(src, pairwise_features, src_mask,return_aw=return_aw)
            #         attention_weights.append(aw)
            #     else:
            #         src,pairwise_features=layer(src, pairwise_features, src_mask,return_aw=return_aw)
            # else:
            #     if return_aw:
            #         src,aw=layer(src, pairwise_features, return_aw=return_aw)
            #         attention_weights.append(aw)
            #     else:
            #         src,pairwise_features=layer(src, pairwise_features, return_aw=return_aw)
            src,pairwise_features=layer(src, pairwise_features, src_mask,return_aw=return_aw,use_gradient_checkpoint=self.use_gradient_checkpoint)
            #print(src.shape)
        #output = self.decoder(src).squeeze(-1)+pairwise_features.mean()*0


        return src, pairwise_features
    
class TriangleAttention(nn.Module):
    def __init__(self, in_dim=128, dim=32, n_heads=4, wise='row'):
        super(TriangleAttention, self).__init__()
        self.n_heads = n_heads
        self.wise = wise
        self.norm = nn.LayerNorm(in_dim)
        self.to_qkv = nn.Linear(in_dim, dim * 3 * n_heads, bias=False)
        self.linear_for_pair = nn.Linear(in_dim, n_heads, bias=False)
        self.to_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Sigmoid()
        )
        self.to_out = nn.Linear(n_heads * dim, in_dim)
        # self.to_out.weight.data.fill_(0.)
        # self.to_out.bias.data.fill_(0.)

    def forward(self, z, src_mask):
        """
        how to do masking
        for row tri attention:
        attention matrix is brijh, where b is batch, r is row, h is head
        so mask should be b()ijh, i.e. take self attention mask and unsqueeze(1,-1)
        add negative inf to matrix before softmax

        for col tri attention
        attention matrix is bijlh, so take self attention mask and unsqueeze(3,-1)

        take src_mask and spawn pairwise mask, and unsqueeze accordingly
        """

        #spwan pair mask
        src_mask[src_mask==0]=-1
        src_mask=src_mask.unsqueeze(-1).float()
        attn_mask=torch.matmul(src_mask,src_mask.permute(0,2,1))


        wise = self.wise
        z = self.norm(z)
        q, k, v = torch.chunk(self.to_qkv(z), 3, -1)
        q, k, v = map(lambda x: rearrange(x, 'b i j (h d)->b i j h d', h=self.n_heads), (q, k, v))
        b = self.linear_for_pair(z)
        gate = self.to_gate(z)
        scale = q.size(-1) ** .5
        if wise == 'row':
            eq_attn = 'brihd,brjhd->brijh'
            eq_multi = 'brijh,brjhd->brihd'
            b = rearrange(b, 'b i j (r h)->b r i j h', r=1)
            softmax_dim = 3
            attn_mask=rearrange(attn_mask, 'b i j->b 1 i j 1')
        elif wise == 'col':
            eq_attn = 'bilhd,bjlhd->bijlh'
            eq_multi = 'bijlh,bjlhd->bilhd'
            b = rearrange(b, 'b i j (l h)->b i j l h', l=1)
            softmax_dim = 2
            attn_mask=rearrange(attn_mask, 'b i j->b i j 1 1')
        else:
            raise ValueError('wise should be col or row!')
        logits = (torch.einsum(eq_attn, q, k) / scale + b)
        # plt.imshow(attn_mask[0,0,:,:,0])
        # plt.show()
        # exit()
        logits = logits.masked_fill(attn_mask == -1, float('-1e-9'))
        attn = logits.softmax(softmax_dim)
        # print(attn.shape)
        # print(v.shape)
        out = torch.einsum(eq_multi, attn, v)
        out = gate * rearrange(out, 'b i j h d-> b i j (h d)')
        z_ = self.to_out(out)
        return z_



if __name__ == "__main__":
    import yaml
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            self.entries=entries

        def print(self):
            print(self.entries)

    def load_config_from_yaml(file_path):
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        return Config(**config)
    config = load_config_from_yaml("configs/pairwise.yaml")
    model=RibonanzaNet(config).cuda()
    x=torch.ones(4,128).long().cuda()
    mask=torch.ones(4,128).long().cuda()
    print(model(x,src_mask=mask).shape)

    # tri_attention=TriangleAttention(wise='row')
    # dummy=torch.ones(6,16,16,128)
    # src_mask=torch.ones(6,16)
    # src_mask[:,12:16]=0
    # out=tri_attention(dummy, src_mask, )
    # print(out.shape) 
