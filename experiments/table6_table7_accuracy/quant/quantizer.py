import torch,torch.nn as nn,torch.nn.functional as F,math
from typing import Union,Literal

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,s,zp,qmin,qmax):
        x_int = ((x/s).round() + zp).clip(qmin,qmax)
        return (x_int - zp) * s
    
    def backward(ctx,grad_output):
        return grad_output,None,None,None,None

def round_ste(x:torch.Tensor):
    return (x.round() - x).detach() + x

class Quantizer(nn.Module):
    def __init__(self,bits:int=16,sym=True,groupsize=-1,clip_ratio=1.0,
                 dynamic=True,dynamic_method:Union[Literal["pertoken"],Literal["pertensor"],Literal["perchannel"]]="pertoken",
                 static_calib_method="minmax",mse=False,lwc=False,shape=None,narrow_symmetric=False):
        super(Quantizer,self).__init__()
        self.bits = bits
        self.sym = sym
        self.narrow_symmetric = narrow_symmetric
        self.group_size = groupsize
        self.mse = mse
        self.clip_ratio = clip_ratio 
        if sym:
            self.qmax = (2**(bits - 1)) - 1
            # Original OSTQuant uses the full signed integer range
            # [-(2^(bits-1)), 2^(bits-1)-1].  The rebuttal mixed-precision
            # Ultra-DSP runs enable narrow symmetric quantization, i.e.,
            # [-(2^(bits-1)-1), 2^(bits-1)-1].
            self.qmin = -self.qmax if narrow_symmetric else -(2**(bits - 1))
        else:
            self.qmax,self.qmin = 2**(bits)-1 , 0
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method 
        if self.bits == 16:
            self.dynamic_method == "pertensor"
        
        self.static_calib_method = static_calib_method
        self.is_observing = False
        self.scale = None

        
        self.lwc = lwc
        if self.lwc:
            dim = shape[0]
            if groupsize != -1:
                dim = dim * (shape[-1] % groupsize)
            self.upbound_factor = nn.Parameter(torch.ones(dim,1)*4)
            self.lowbound_factor = nn.Parameter(torch.ones(dim,1,)*4)
    
    def fake_quant(x):
        pass
    
    def forward(self,x):
        if self.bits > 16:
            return x
        x_dtype = x.dtype
        
        if self.dynamic:
            if self.bits >= 16:
                return x
            s,zp = self.dynamic_quant_params(x)
            
            x_fake = ((round_ste(x/s) + zp).clip(self.qmin,self.qmax) - zp)*s
            return x_fake.to(x_dtype)
        
        else:
            s,zp = self.static_quant_params(x)
            if self.is_observing:
                return x
            return STE.apply(x,s,zp,self.qmin,self.qmax).to(x_dtype)
            x_fake = ((round_ste(x/s) + zp).clip(self.qmin,self.qmax) - zp)*s
            return x_fake.to(x_dtype)
            
    
    def quantize_to_int(self, x: torch.Tensor):
        if self.bits >= 16:
            raise ValueError("quantize_to_int only supports quantizers with bits < 16")

        if self.dynamic:
            s, zp = self.dynamic_quant_params(x)
        else:
            s, zp = self.static_quant_params(x)

        q = (torch.round(x / s) + zp).clip(self.qmin, self.qmax)
        return q.to(torch.int8), s, zp
    
    
    def dynamic_quant_params(self,x):
        ori_shape = list(x.shape)
        if self.dynamic_method == "pertensor":
            xmax,xmin = torch.max(x),torch.min(x)
        elif self.dynamic_method == "pertoken" or self.group_size != -1:
            if self.group_size != -1:
                reshaped_x = x.reshape(*ori_shape[:-1],ori_shape[-1] // self.group_size, self.group_size)
                xmax,xmin = torch.amax(reshaped_x,dim=-1,keepdim=True),torch.amin(reshaped_x,dim=-1,keepdim=True)
            else:
                xmax,xmin = torch.amax(x,dim=tuple(range(min(2,x.dim()-1),x.dim())),keepdim=True),torch.amin(x,dim=tuple(range(min(2,x.dim()-1),x.dim())),keepdim=True) 
        elif self.dynamic_method == "perchannel": 
            xmax,xmin = torch.amax(x,dim=list(range(0,x.dim()-1)),keepdim=True),torch.amin(x,dim=(0,1),keepdim=False)
        else:
            raise NotImplemented
        if self.lwc:
            xmax,xmin = F.sigmoid(self.upbound_factor)*xmax , F.sigmoid(self.lowbound_factor) * xmin
        xmax = xmax.clip(min=1e-6)
        xmin = xmin.clip(max=0)
        xmax,xmin = xmax*self.clip_ratio, xmin * self.clip_ratio
        if self.mse:
            x = x.detach()
            xmax = xmax.detach()
            xmin = xmin.detach()
            scale= torch.ones_like(xmax)
            zp = torch.zeros_like(xmax)
            best_err = torch.ones_like(xmax).fill_(10000)
            for i in range(80):
                p = 1 - (i/100)
                xmin1 = p * xmin.detach()
                xmax1 = p * xmax.detach()
                if self.sym:
                    s1 = torch.maximum(xmax1/self.qmax ,xmin1/self.qmin)
                    zp1 = torch.zeros_like(s1)
                else:
                    s1 = (xmax1 - xmin1) / self.qmax
                    zp1 = torch.round(-xmin1 / s1)
                q = (((x/s1).round() + zp1).clip(self.qmin,self.qmax) - zp1) * s1 
                err = q.sub_(x.data).pow_(2).sum(-1,keepdim=True)
                tmp = err < best_err
                if torch.any(tmp):
                    scale[tmp] = s1[tmp].to(scale.dtype)
                    zp[tmp] = zp1[tmp].to(zp.dtype)
                    best_err[tmp] = err[tmp].to(best_err.dtype)
        else:
            if self.sym:
                scale = torch.maximum((xmin)/self.qmin ,xmax/(self.qmax))
                
                zp = torch.zeros_like(scale)
            else:
                scale = (xmax - xmin).clip_(1e-6) / self.qmax
                zp = torch.round(-xmin / scale)

        if self.group_size != -1:
            scale = scale.repeat(1, 1, 1, self.group_size).reshape(ori_shape)
            zp = zp.repeat(1, 1, 1, self.group_size).reshape(ori_shape)
        return scale, zp 
    
    def percent(self,x,percent:float):
        if self.dynamic_method == "pertensor":
            return torch.quantile(x.abs(),percent)
        elif self.dynamic_method == "perchannel":
            return torch.quantile(x.abs().reshape(-1,x.shape[-1]),percent,dim=0)
        else:
            raise NotImplemented
    
    def static_quant_params(self,x):
        if not self.is_observing:
            assert self.scale is not None,"must be set before static quantization"
            return self.scale,self.zp
        assert self.sym == True,"only support"
        if self.static_calib_method == "minmax":
            if self.scale is None:
                self.scale,self.zp  = self.dynamic_quant_params(x)
            else:
                scale,self.zp = self.dynamic_quant_params(x)
                self.scale = torch.max(self.scale,scale)
        elif "percent" in self.static_calib_method:
            percent = float(self.static_calib_method[7:])
            if self.scale is None:
                self.scale = self.percnet(x,percent)
                self.zp = torch.zeros_like(self.scale)
            else:
                scale = self.percent(x,percent)
                self.scale = scale * 0.01 + self.scale * 0.99
        return self.scale,self.zp
            
    def enable_dynamic(self,value):
        self.dynamic = value

    def enable_observer(self,value=True):
        self.dynamic = False
        self.is_observing = value
