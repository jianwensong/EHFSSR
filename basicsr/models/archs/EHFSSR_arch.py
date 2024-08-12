import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange
import numbers
import math
# from timm.models.layers import trunc_normal_, DropPath

class EHFSSR(nn.Module):
    def __init__(self, up_scale=4, dim=64, groups=5,num=6):
        super(EHFSSR, self).__init__()
        self.init = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        self.num = num
        self.body = nn.ModuleList()
        self.groups = groups
        for i in range(groups):
            self.body.append(GroupSR(dim,num=num))
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3 * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.PixelShuffle(up_scale)
        )
        self.up_scale = up_scale
        self.fuse = nn.Sequential(nn.Conv2d(dim,dim,1),eca_layer(dim,3))
    def forward(self, fea):
        x_left0,x_right0  = fea.chunk(2, dim=1)
        x_left = self.init(x_left0)
        x_right = self.init(x_right0)
        x_left0 = x_left
        x_right0 = x_right
        for i in range(self.groups):
            x_left,x_right  = self.body[i](x_left,x_right)
        # x_left = self.fuse(x_left)
        # x_right = self.fuse(x_right)
        x_left = self.fuse(x_left) + x_left0
        x_right = self.fuse(x_right) + x_right0
        x_left = self.up(x_left)#+F.interpolate(x_left0, scale_factor=self.up_scale, mode='bicubic')
        x_right = self.up(x_right)#+F.interpolate(x_right0, scale_factor=self.up_scale, mode='bicubic')
        return torch.cat([x_left,x_right],dim=1)

class GroupSR(nn.Module):
    def __init__(self, dim, num=6):
        super().__init__()
        self.num = num
        self.body = nn.ModuleList()
        for i in range(num):
            self.body.append(BasicBlockSR(dim,interaction=(i==0),selfatn =(i!=0),window_sizes = [8,16],shift=(i%2==0),gsa=(i==(num-1))))
        # self.conv =  nn.Conv2d(dim,dim,3,1,1)
        self.conv = nn.Sequential(nn.Conv2d(dim,dim,1),eca_layer(dim,3))
    def forward(self,x_left0,x_right0):
        x_left,x_right=x_left0,x_right0
        for i in range(self.num):
            x_left,x_right = self.body[i](x_left,x_right)
        x_left = self.conv(x_left)
        x_right = self.conv(x_right)
        return x_left+x_left0,x_right+x_right0

class BasicBlockSR(nn.Module):
    def __init__(self, dim, interaction=False, selfatn=False,window_sizes=[4,8,16],shift=False, gsa = False):
        super().__init__()
        if selfatn & (not gsa):
            self.selfatn = WSAttention(dim,window_sizes = [16,16],down_rates = [1,2], shift=shift)
        if selfatn & gsa:
            self.selfatn = GSAttention(dim,window_sizes = [16,16],down_rates = [1,1])
        if not selfatn:
            self.selfatn = None
        self.gsa = gsa
        self.interaction = Crossattention(dim,window_sizes=[1,1]) if interaction else None
        self.window_sizes = window_sizes
        self.MLP = MLP(dim,ratio=2)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def check_image_size(self, x, window_sizes):
        _, _, h, w = x.size()
        wsize = window_sizes[0]
        for i in range(1, len(window_sizes)):
            wsize = wsize*window_sizes[i] // math.gcd(wsize, window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, x_left,x_right):
        b,c,h,w=x_left.shape
        x_left = self.MLP(x_left)
        x_right = self.MLP(x_right)
        if self.selfatn:
            if not self.gsa:
                x_left = self.check_image_size(x_left,self.window_sizes)
                x_right = self.check_image_size(x_right,self.window_sizes)
                x_left = self.selfatn(x_left)[:,:,:h,:w]
                x_right = self.selfatn(x_right)[:,:,:h,:w]
            else:
                x_left = self.selfatn(x_left)
                x_right = self.selfatn(x_right)
        if self.interaction:
            x_left,x_right = self.interaction(x_left,x_right)
        return x_left,x_right

class Crossattention(nn.Module):
    def __init__(self,dim,window_sizes=[1, 1, 2, 4],shift=False):
        super(Crossattention,self).__init__()
        self.LayernormL = LayerNorm2d(dim)
        self.LayernormR = LayerNorm2d(dim)
        self.feaL = nn.Conv2d(dim,dim,1,bias=False)
        self.feaR = nn.Conv2d(dim,dim,1,bias=False)
        self.to_l = nn.Conv2d(dim,dim,1,bias=False)
        self.to_r = nn.Conv2d(dim,dim,1,bias=False)
        self.transition = nn.Sequential(nn.Conv2d(dim,dim,3,1,1,groups=dim),nn.Conv2d(dim,dim,1))
        self.softmax = nn.Softmax(dim=-1)
        self.window_sizes = window_sizes
        self.shift = shift
        self.out = nn.Conv2d(dim,dim,1,bias=False)
        self.scale = (dim//len(window_sizes)) ** -0.5
    def shifts(self,x,n_div):
        b,c,h,w = x.shape
        g = c // n_div
        out = torch.zeros_like(x)
        out[:, g * 0:g * 1, :, :-1] = x[:, g * 0:g * 1, :, 1:]  # shift left
        out[:, g * 1:g * 2, :, 1:] = x[:, g * 1:g * 2, :, :-1]  # shift right
        out[:, g * 2:g * 3, :-1, :] = x[:, g * 2:g * 3, 1:, :]  # shift up
        out[:, g * 3:g * 4, 1:, :] = x[:, g * 3:g * 4, :-1, :]  # shift down
        out[:, g * 4:, :, :] = x[:, g * 4:, :, :]  # no shift
        return out
    def forward(self,xl0,xr0):
        B,C,H,W= xl0.shape
        vl = self.feaL(self.shifts(xl0,8))
        vr = self.feaR(self.shifts(xr0,8))
        x_left = self.transition(xl0)+xl0
        x_right = self.transition(xr0)+xr0
        x_left = self.LayernormL(x_left)
        x_left = self.to_l(self.shifts(x_left,8))
        x_right = self.LayernormR(x_right)
        x_right = self.to_r(self.shifts(x_right,8))
        x_left = x_left.chunk(len(self.window_sizes),dim=1)
        x_right = x_right.chunk(len(self.window_sizes),dim=1)
        vls = vl.chunk(len(self.window_sizes),dim=1)
        vrs = vr.chunk(len(self.window_sizes),dim=1)
        warpLs=[]
        warpRs=[]
        maxdisp = 192//2
        mask = (torch.tril(torch.ones(W,W),diagonal=0)*torch.triu(torch.ones(W,W),diagonal=-maxdisp)).repeat(B*H,1,1).to(xl0.device)
        for idx in range(len(self.window_sizes)):
            xl = x_left[idx]
            xr = x_right[idx]
            vl = vls[idx]
            vr = vrs[idx]
            xl = rearrange(xl,'b c h w-> (b h) w c')
            xr = rearrange(xr,'b c h w-> (b h) w c')
            vl = rearrange(vl,'b c h w-> (b h) w c')
            vr = rearrange(vr,'b c h w-> (b h) w c')
            atn = torch.bmm(xl,xr.permute(0,2,1)) * mask
            warpL = torch.bmm(atn.softmax(dim=-1),vr)
            warpR = torch.bmm(atn.permute(0,2,1).softmax(dim=-1),vl)
            warpL = rearrange(warpL,'(b h) w c->b c h w',h=H)
            warpR = rearrange(warpR,'(b h) w c->b c h w',h=H)
            warpLs.append(warpL)
            warpRs.append(warpR)
        xl = torch.cat(warpLs,dim=1)
        xr = torch.cat(warpRs,dim=1)
        xl = self.out(xl)
        xr = self.out(xr)
        return xl+xl0,xr+xr0

class WSAttention(nn.Module):
    def __init__(self,channels,window_sizes = [8,8],down_rates = [1,2],shift = False):
        super().__init__()
        self.Layernorm = LayerNorm2d(channels)    
        self.channels = channels
        self.window_sizes = window_sizes
        split_channel = int(channels//len(window_sizes))
        self.depthwise  = nn.Conv2d(channels,channels,3,1,1,groups=channels)
        self.CA = effCA(channels,3)
        self.to_q = nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1,bias=False)
        self.scale = split_channel ** -0.5
        self.down = nn.Conv2d(split_channel,split_channel,kernel_size=2, stride=2,groups=split_channel)
        self.down_rates = down_rates
        self.sa = nn.Sequential(nn.Conv2d(channels,1,1),nn.Sigmoid())
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.shift = shift
        # self.depthbranch = nn.Conv2d(channels, channels, kernel_size=1)
    def forward(self, x0):
        b,c,h,w = x0.shape
        x = self.Layernorm(x0)
        x1 = self.depthwise(x)
        vs = (self.to_v(x) * self.CA(x1)).chunk(len(self.window_sizes),dim=1)
        # vs = vs.chunk(len(self.window_sizes),dim=1)
        qs = self.to_q(x).chunk(len(self.window_sizes),dim=1)
        ks = self.to_k(x).chunk(len(self.window_sizes),dim=1)
        ys = []
        for idx in range(len(self.window_sizes)):
            wsize = self.window_sizes[idx]
            downsize = wsize//self.down_rates[idx]
            q = qs[idx]
            if self.shift:
                q=torch.roll(q, shifts=(-wsize//2, -wsize//2), dims=(2,3))
            q = rearrange(q, 'b c (h dh) (w dw) -> (b h w) (dh dw) c',  dh=wsize, dw=wsize)
            if self.down_rates[idx]!=1:
                k = self.down(ks[idx])
                v = self.down(vs[idx])
                if self.shift:
                    k = torch.roll(k, shifts=(-downsize//2, -downsize//2), dims=(2,3))
                    v = torch.roll(v, shifts=(-downsize//2, -downsize//2), dims=(2,3))
                k = rearrange(k, 'b c (h dh) (w dw) -> (b h w) (dh dw) c',  dh=downsize, dw=downsize)
                v = rearrange(v, 'b c (h dh) (w dw) -> (b h w) (dh dw) c',  dh=downsize, dw=downsize)
            else:
                k = ks[idx]
                v = vs[idx]
                if self.shift:
                    k = torch.roll(k, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                    v = torch.roll(v, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                k = rearrange(k, 'b c (h dh) (w dw) -> (b h w) (dh dw) c',  dh=wsize, dw=wsize)
                v = rearrange(v, 'b c (h dh) (w dw) -> (b h w) (dh dw) c',  dh=wsize, dw=wsize)
            atn = (q @ k.transpose(-2, -1)) * self.scale 
            atn = atn.softmax(dim=-1)
            y_ = (atn @ v)
            y_ = rearrange(
                y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
            )
            if self.shift:
                y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
            ys.append(y_)
        y = torch.cat(ys, dim=1)   
        x1 =  self.sa(y)*x1
        y = self.project_out(y+x1)
        return y + x0

class GSAttention(nn.Module):
    def __init__(self,channels,window_sizes = [8,8],down_rates = [1,2]):
        super().__init__()
        self.Layernorm = LayerNorm2d(channels)
        dim = channels//2    
        self.to_qkv = nn.Conv2d(channels, dim*3, kernel_size=1,bias=False)
        self.to_sp = nn.Conv2d(channels, dim, kernel_size=1,bias=False)
        self.conv = nn.Conv2d(dim,dim,kernel_size=9,stride=1,padding=4,groups=dim)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.out = nn.Conv2d(channels,channels,kernel_size=1,bias=False)
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
    def forward(self, x0):
        b,c,h,w = x0.shape
        x = self.Layernorm(x0)
        qkv = self.to_qkv(x)
        q,k,v = rearrange(qkv, 'b (qkv c) h w -> qkv b c (h w)',qkv=3)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out1 = (attn @ v)
        
        out1 = rearrange(out1, 'b c (h w) -> b c h w', h=h, w=w)

        out2 = self.to_sp(x)
        out2 = self.relu(self.conv(out2))
        out = self.out(torch.cat([out1,out2],dim=1))
        return out + x0

class MLP(nn.Module):
    def __init__(self, dim,ratio=2):
        super(MLP,self).__init__()
        super().__init__()
        self.layernorm1 = LayerNorm2d(dim)
        expandim = int(dim*ratio)
        self.proj1 = nn.Conv2d(dim,expandim,1)
        self.conv = nn.Conv2d(expandim,expandim,5,1,2,groups=expandim)
        self.projout = nn.Conv2d(expandim,dim,1)
        self.ca = eca_layer(dim,3)
        self.act = nn.LeakyReLU(0.1,inplace=True)
    def forward(self,x0):
        x = self.layernorm1(x0)
        x = self.proj1(x)
        x = self.conv(x) + x
        x = self.projout(self.act(x))
        x = self.ca(x)
        return x + x0

class LPE(nn.Module):
    def __init__(self, dim):
        super(LPE, self).__init__()
        self.DWConv = nn.Conv2d(dim, dim,3,1,1,groups=dim)

    def forward(self, x):
        result = self.DWConv(x) + x
        return result

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=1, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=2, dilation=2,bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=k_size, padding=4, dilation=4,bias=False)
        self.conv = nn.Conv1d(3, 1, kernel_size=k_size, padding=1, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        b,c,h,w = x.shape
        y = self.avg_pool(x)
        y = y.squeeze(-1).permute(0,2,1)
        # Two different branches of ECA module
        y1 = self.conv1(y)
        y2 = self.conv2(y)
        y3 = self.conv3(y)
        # Multi-scale information fusion
        y= self.conv(torch.cat([y1,y2,y3],dim=1))
        y = self.sigmoid(y.permute(0,2,1).unsqueeze(-1))
        return x * y

class effCA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(effCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=1, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=2, dilation=2,bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=k_size, padding=4, dilation=4,bias=False)
        self.conv = nn.Conv1d(3, 1, kernel_size=k_size, padding=1, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        b,c,h,w = x.shape
        y = self.avg_pool(x)
        y = y.squeeze(-1).permute(0,2,1)
        # Two different branches of ECA module
        y1 = self.conv1(y)
        y2 = self.conv2(y)
        y3 = self.conv3(y)
        # Multi-scale information fusion
        y= self.conv(torch.cat([y1,y2,y3],dim=1))
        y = self.sigmoid(y.permute(0,2,1).unsqueeze(-1))
        return y

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, gumbels

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps
    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

if __name__ == "__main__":
    net = EHFSSR(up_scale=4,  dim=64, groups=5,num=6)
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
