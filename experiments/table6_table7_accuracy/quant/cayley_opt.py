import torch,torch.nn as nn,torch.nn.functional as F,random,numpy as np
from torch.optim.optimizer import Optimizer,required
from torch.optim import SGD

def Cayley_loop(X, W, tan_vec, t): 
    [n, p] = X.size()
    Y = X + t * tan_vec
    for i in range(5):
        Y = X + t * torch.matmul(W, 0.5*(X+Y))
    return Y.t()

def unit(v, dim=1, eps=1e-8):
    vnorm = norm(v, dim)
    return v/vnorm.add(eps), vnorm

def qr_retraction(tan_vec): 
    [p,n] = tan_vec.size()
    tan_vec.t_()
    q,r = torch.qr(tan_vec)
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)
    q.t_()
    return q

def norm(v, dim=1):
    assert len(v.size())==2
    return v.norm(p=2, dim=dim, keepdim=True)

def matrix_norm_one(W):
    out = torch.abs(W)
    out = torch.sum(out, dim=0)
    out = torch.max(out)
    return out

episilon = 1e-8

class SGDG(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 stiefel=False, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        stiefel=stiefel, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault("maximize",False)
            group.setdefault("foreach",None)
            group.setdefault("differentiable",False)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            stiefel = group['stiefel']
                           
            for p in group['params']:
                if p.grad is None:
                    continue
                weight_decay = group['weight_decay']
                dampening = group['dampening']
                nesterov = group['nesterov']

                unity,_ = unit(p.data.view(p.size()[0],-1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']
                    
                    rand_num = random.randint(1,101)
                    if rand_num==1:
                        unity = qr_retraction(unity)
                    
                    g = p.grad.data.view(p.size()[0],-1)
                       
                    
                    lr = group['lr']
                    
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state: 
                        
                        
                        
                        param_state['momentum_buffer'] = torch.zeros_like(g.t())
                            
                    V = param_state['momentum_buffer']
                    V = momentum * V - g.t()   
                    MX = torch.mm(V, unity)
                    XMX = torch.mm(unity, MX)
                    XXMX = torch.mm(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()
                    t = 0.5 * 2 / (matrix_norm_one(W) + episilon)                    
                    alpha = min(t, lr)
                    
                    p_new = Cayley_loop(unity.t(), W, V, alpha)
                    V_new = torch.mm(W, unity.t()) 

                    p.data.copy_(p_new.view(p.size()))
                    
                    param_state['momentum_buffer'].copy_(V_new)
                    

                else:
                    d_p = p.grad.data
                    weight_decay = group['weight_decay']
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss

class AdamG(Optimizer):

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, 
                 grassmann=False, beta2=0.99, epsilon=1e-8, omega=0, grad_clip=None):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, 
                        grassmann=grassmann, beta2=beta2, epsilon=epsilon, omega=0, grad_clip=grad_clip)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AdamG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            stiefel = group['stiefel']
            
            for p in group['params']:
                if p.grad is None:
                    continue
            
                beta1 = group['momentum']
                beta2 = group['beta2']
                epsilon = group['epsilon']

                unity,_ = unit(p.data.view(p.size()[0],-1))
                if stiefel and unity.size()[0] <= unity.size()[1]:
                    rand_num = random.randint(1,101)
                    if rand_num==1:
                        unity = qr_retraction(unity)
                        
                    g = p.grad.data.view(p.size()[0],-1)

                    param_state = self.state[p]
                    if 'm_buffer' not in param_state:
                        size=p.size()
                        param_state['m_buffer'] = torch.zeros([int(np.prod(size[1:])), size[0]])
                        param_state['v_buffer'] = torch.zeros([1])
                        if p.is_cuda:
                            param_state['m_buffer'] = param_state['m_buffer'].cuda()
                            param_state['v_buffer'] = param_state['v_buffer'].cuda()

                        param_state['beta1_power'] = beta1
                        param_state['beta2_power'] = beta2

                    m = param_state['m_buffer']
                    v = param_state['v_buffer']
                    beta1_power = param_state['beta1_power']
                    beta2_power = param_state['beta2_power']

                    mnew = beta1*m  + (1.0-beta1)*g.t() 
                    vnew = beta2*v  + (1.0-beta2)*(torch.norm(g)**2)
                    
                    mnew_hat = mnew / (1 - beta1_power)
                    vnew_hat = vnew / (1 - beta2_power)
                    
                    MX = torch.matmul(mnew_hat, unity)
                    XMX = torch.matmul(unity, MX)
                    XXMX = torch.matmul(unity.t(), XMX)
                    W_hat = MX - 0.5 * XXMX
                    W = (W_hat - W_hat.t())/vnew_hat.add(epsilon).sqrt()
                    
                    t = 0.5 * 2 / (matrix_norm_one(W) + episilon)                    
                    alpha = min(t, group['lr'])
                    
                    p_new = Cayley_loop(unity.t(), W, mnew, -alpha)

                    p.data.copy_(p_new.view(p.size()))
                    mnew = torch.matmul(W, unity.t()) * vnew_hat.add(epsilon).sqrt() * (1 - beta1_power)
                    m.copy_(mnew)
                    v.copy_(vnew)

                    param_state['beta1_power']*=beta1
                    param_state['beta2_power']*=beta2
                    
                else:
                    momentum = group['momentum']
                    weight_decay = group['weight_decay']
                    dampening = group['dampening']
                    nesterov = group['nesterov']
                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

        return loss       
