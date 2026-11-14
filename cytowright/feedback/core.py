"""Construct parameter updates for cellular-response predictors."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from .basis import fit_basis
from .linear import solve_ridge as ridge_solve


class FeedbackModel(nn.Module):
    def __init__(self,base,context_mode,context_dim,rank=32,target_dim=1678):
        super().__init__();self.base=base;self.context_mode=context_mode
        self.chemical_basis=nn.Parameter(torch.zeros(2048,rank),requires_grad=False)
        self.compound_readout=nn.Linear(rank,target_dim,bias=False);nn.init.zeros_(self.compound_readout.weight)
        for name,value in [('gate_projection',torch.zeros(context_dim,rank)),('gate_center',torch.zeros(rank)),
                           ('gate_scale',torch.ones(rank)),('chemical_scale',torch.ones(rank))]:
            self.register_buffer(name,value)

    def context(self,pre):
        return self.base.control_encoder(pre) if self.context_mode=='control_encoder' else pre

    def compound_features(self,pre,chemical):
        gate=1+.25*torch.tanh((self.context(pre)@self.gate_projection-self.gate_center)/self.gate_scale)
        return (chemical@self.chemical_basis/self.chemical_scale)*gate

    def forward(self,pre,chemical,dose):
        return self.base(pre,chemical,dose)+self.compound_readout(self.compound_features(pre,chemical))


def normalized_inputs(tensors,kind):
    s=tensors.standardizer
    return s.normalize_pre(getattr(tensors,kind+'_pre')),s.normalize_condition(getattr(tensors,kind+'_condition'))


def predict_normalized(model,pre,condition):
    rows=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(pre),256):
            y=model(pre[start:start+256],condition[start:start+256,:2048],condition[start:start+256,2048:])
            if y.ndim!=2 or len(y)!=min(256,len(pre)-start) or not torch.isfinite(y).all():raise ValueError('Prediction interface failed')
            rows.append(y.detach().numpy())
    return np.concatenate(rows)


def predict_raw(model,tensors,pre,condition):
    s=tensors.standardizer
    p=s.normalize_pre(torch.as_tensor(pre,dtype=torch.float32))
    c=s.normalize_condition(torch.as_tensor(condition,dtype=torch.float32))
    return s.inverse_target(torch.from_numpy(predict_normalized(model,p,c))).numpy()


def metrics(p,y,cp_dim=701):
    p=p.astype(np.float64);y=y.astype(np.float64)
    return {'global_pcc':float(np.corrcoef(p.ravel(),y.ravel())[0,1]),'mse':float(np.mean((p-y)**2)),
            'cp_pcc':float(np.corrcoef(p[:,:cp_dim].ravel(),y[:,:cp_dim].ravel())[0,1]),
            'l1000_pcc':float(np.corrcoef(p[:,cp_dim:].ravel(),y[:,cp_dim:].ravel())[0,1])}


def affine_interfaces(model,pre,condition):
    """Select terminal affine modules by exact prediction identity, using Fit inputs."""
    seen=[];hooks=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear):
            hooks.append(module.register_forward_hook(lambda m,args,out,name=name:seen.append((name,args[0].detach().clone(),out.detach().clone()))))
    with torch.inference_mode():out=model(pre[:8],condition[:8,:2048],condition[:8,2048:])
    for hook in hooks:hook.remove()
    def matching(start,end):
        return [name for name,x,y in seen if x.ndim==2 and y.shape==out[:,start:end].shape
                and torch.equal(y,out[:,start:end]) and sum(v[0]==name for v in seen)==1]
    target_dim=out.shape[1];cp_dim=pre.shape[1]
    joint=matching(0,target_dim)
    if joint:return [(joint[-1],0,target_dim)]
    cp,ge=matching(0,cp_dim),matching(cp_dim,target_dim)
    if cp and ge and cp[-1]!=ge[-1]:return [(cp[-1],0,cp_dim),(ge[-1],cp_dim,target_dim)]
    return []


def capture_affine_features(model,pre,condition,interfaces):
    rows={name:[] for name,_,_ in interfaces};hooks=[]
    for name in rows:
        hooks.append(model.get_submodule(name).register_forward_pre_hook(
            lambda m,args,name=name:rows[name].append(args[0].detach().numpy().astype(np.float64))))
    baseline=predict_normalized(model,pre,condition)
    for hook in hooks:hook.remove()
    result={name:np.concatenate(parts) for name,parts in rows.items()}
    if any(len(x)!=len(pre) for x in result.values()):raise ValueError('Terminal module reused across calls')
    return result,baseline


def output_proposal(parent,features,baseline,target,interfaces,lam):
    candidate=copy.deepcopy(parent).eval()
    with torch.no_grad():
        for name,start,end in interfaces:
            head=candidate.get_submodule(name);f=features[name]
            design=np.column_stack([f,np.ones(len(f))]) if head.bias is not None else f
            delta=ridge_solve(design,target[:,start:end]-baseline[:,start:end],lam)
            if head.bias is not None:
                head.weight.add_(torch.tensor(delta[:-1].T,dtype=torch.float32));head.bias.add_(torch.tensor(delta[-1],dtype=torch.float32))
            else:head.weight.add_(torch.tensor(delta.T,dtype=torch.float32))
    return candidate


def context_interface(base,pre):
    # Shape-only host choice; no target or development score decides this mode.
    if hasattr(base,'control_encoder') and callable(base.control_encoder):
        try:
            with torch.inference_mode():h=base.control_encoder(pre[:8])
            if h.ndim==2 and len(h)==min(8,len(pre)) and torch.isfinite(h).all():return 'control_encoder',h.shape[1]
        except (TypeError,RuntimeError,AttributeError):pass
    return 'pre',pre.shape[1]


def route_proposals_base(base,tensors,seed):
    pre,condition=normalized_inputs(tensors,'fit');mode,width=context_interface(base,pre)
    result=FeedbackModel(copy.deepcopy(base),mode,width,target_dim=tensors.fit_target.shape[1]).eval()
    baseline=predict_normalized(base,pre,condition).astype(np.float64)
    target=tensors.standardizer.normalize_target(tensors.fit_target).numpy().astype(np.float64)
    residual=target-baseline;chemical=condition[:,:2048].numpy().astype(np.float64)
    basis=fit_basis(chemical,residual,32,seed)
    rng=np.random.default_rng(seed+417);projection=rng.normal(size=(width,32))/np.sqrt(width)
    with torch.inference_mode():context=result.context(pre).numpy().astype(np.float64)
    g=context@projection;center=g.mean(0);scale=np.maximum(g.std(0),1e-6)
    cscale=np.maximum(np.sqrt(np.mean((chemical@basis)**2,axis=0)),1e-6)
    with torch.no_grad():
        for name,value in [('chemical_basis',basis),('gate_projection',projection),('gate_center',center),('gate_scale',scale),('chemical_scale',cscale)]:
            getattr(result,name).copy_(torch.tensor(value,dtype=torch.float32))
        features=result.compound_features(pre,condition[:,:2048]).numpy().astype(np.float64)
    return result,features,residual,{'mode':mode,'dimension':width,'rank':32}


def interface_check():
    class Single(nn.Module):
        def __init__(self):super().__init__();self.readout=nn.Linear(701,1678)
        def forward(self,p,c,d):return self.readout(p)
    class Split(nn.Module):
        def __init__(self):super().__init__();self.cp=nn.Linear(701,701);self.ge=nn.Linear(701,977)
        def forward(self,p,c,d):return torch.cat([self.cp(p),self.ge(p)],dim=1)
    class Nonlinear(Single):
        def forward(self,p,c,d):return torch.tanh(self.readout(p))
    torch.manual_seed(181);p=torch.randn(12,701);c=torch.randn(12,2049)
    found=[]
    for cls,expected in [(Single,1),(Split,2),(Nonlinear,0)]:
        m=cls().eval();interfaces=affine_interfaces(m,p,c)
        assert len(interfaces)==expected
        wrapper=FeedbackModel(m,'pre',701).eval()
        with torch.inference_mode():assert torch.equal(m(p,c[:,:2048],c[:,2048:]),wrapper(p,c[:,:2048],c[:,2048:]))
        found.append({'model':cls.__name__,'affine_interfaces':len(interfaces),'zero_route_preserves_native_output':True})
    return {'status':'pass','synthetic_interfaces':found}
