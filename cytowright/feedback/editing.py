"""Fit readout and compound-route corrections from training and validation arrays."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import types
import numpy as np
import torch
from .core import (FeedbackModel,normalized_inputs,predict_raw,metrics,affine_interfaces,
                   capture_affine_features,output_proposal,route_proposals_base,ridge_solve)


class Standardizer:
    def __init__(self,pre,condition,target):
        for name,value in [('pre',pre),('condition',condition),('target',target)]:
            mean=value.mean(0,keepdim=True);scale=value.std(0,unbiased=False,keepdim=True)
            setattr(self,name+'_mean',mean);setattr(self,name+'_scale',torch.where(scale>1e-8,scale,torch.ones_like(scale)))
        h=hashlib.sha256()
        for name in ['pre_mean','pre_scale','condition_mean','condition_scale','target_mean','target_scale']:
            a=getattr(self,name).numpy();h.update(str(tuple(a.shape)).encode('ascii'));h.update(a.tobytes())
        self.fingerprint=h.hexdigest()
    def normalize_pre(self,x):return (x-self.pre_mean)/self.pre_scale
    def normalize_condition(self,x):return (x-self.condition_mean)/self.condition_scale
    def normalize_target(self,x):return (x-self.target_mean)/self.target_scale
    def inverse_target(self,x):return x*self.target_scale.to(x)+self.target_mean.to(x)


def prepare(fit,selection):
    values={}
    for name,part in [('fit',fit),('selection',selection)]:
        for key in ['pre','condition','target']:
            a=np.asarray(part[key],dtype=np.float32)
            if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Finite two-dimensional arrays required')
            values[name+'_'+key]=torch.as_tensor(a)
        n=len(values[name+'_pre'])
        dimensions=(values[name+'_pre'].shape[1],values[name+'_target'].shape[1])
        if values[name+'_condition'].shape!=(n,2049) or dimensions not in [(591,1568),(701,1678)]:
            raise ValueError('Registered chemical/dose and joint-response dimensions required')
    if values['fit_pre'].shape[1:]!=values['selection_pre'].shape[1:] or values['fit_target'].shape[1:]!=values['selection_target'].shape[1:]:
        raise ValueError('Fit and development dimensions differ')
    values['standardizer']=Standardizer(values['fit_pre'],values['fit_condition'],values['fit_target'])
    return SimpleNamespace(**values)


def refine(parent,tensors,selection,seed=41):
    """Evaluate four native output and four compound-route proposals."""
    parent=copy.deepcopy(parent).cpu().eval();pre,condition=normalized_inputs(tensors,'fit')
    target=tensors.standardizer.normalize_target(tensors.fit_target).numpy().astype(np.float64)
    def score(m):return metrics(predict_raw(m,tensors,selection['pre'],selection['condition']),selection['target'],cp_dim=pre.shape[1])
    original=score(parent);best=copy.deepcopy(parent);best_metrics=original;best_id='original';trace=[]
    interfaces=affine_interfaces(parent,pre,condition)
    features,baseline=capture_affine_features(parent,pre,condition,interfaces)
    for lam in [.1,1.,10.,100.]:
        candidate=output_proposal(parent,features,baseline,target,interfaces,lam);value=score(candidate)
        eligible=value['mse']<=original['mse']+.0005;name='output_'+str(lam)
        trace.append({'id':name,'metrics':value,'eligible':eligible})
        if eligible and value['global_pcc']>best_metrics['global_pcc']:best=candidate;best_metrics=value;best_id=name
    route,features,residual,context=route_proposals_base(best,tensors,seed)
    final=FeedbackModel(copy.deepcopy(best),context['mode'],context['dimension'],target_dim=target.shape[1]).eval()
    final_metrics=best_metrics;final_id=best_id
    for lam in [.1,1.,10.,100.]:
        candidate=copy.deepcopy(route)
        with torch.no_grad():candidate.compound_readout.weight.copy_(torch.tensor(ridge_solve(features,residual,lam).T,dtype=torch.float32))
        value=score(candidate);eligible=value['mse']<=original['mse']+.0005;name='compound_'+str(lam)
        trace.append({'id':name,'metrics':value,'eligible':eligible})
        if eligible and value['global_pcc']>final_metrics['global_pcc']:final=candidate;final_metrics=value;final_id=name
    return final,{'selected_candidate':final_id,'original_metrics':original,'selected_metrics':final_metrics,
                  'context':context,'candidates':trace,'seed':seed,'proposal_count':8,
                  'standardization_fingerprint':tensors.standardizer.fingerprint}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True,help='Predictor Python source with build_model(task_spec)')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fit',type=Path,required=True,help='Training NPZ containing pre, condition and target arrays')
    p.add_argument('--selection',type=Path,required=True,help='Validation NPZ containing pre, condition and target arrays')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=41)
    args=p.parse_args(argv);torch.set_num_threads(2)
    if args.output.exists():raise ValueError('Choose a new output directory')
    with np.load(args.fit,allow_pickle=False) as z:fit={k:z[k] for k in ['pre','condition','target']}
    with np.load(args.selection,allow_pickle=False) as z:selection={k:z[k] for k in ['pre','condition','target']}
    tensors=prepare(fit,selection);source=args.source.read_text();module=types.ModuleType('supplied_predictor')
    exec(compile(source,str(args.source),'exec'),module.__dict__)
    parent=module.build_model({'context_dim':fit['pre'].shape[1],'chemical_dim':2048,'dose_dim':1,
                              'target_dim':fit['target'].shape[1],'cp_dim':fit['pre'].shape[1],'l1000_dim':977})
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    fp=payload.get('standardization_fingerprint')
    if fp and fp!=tensors.standardizer.fingerprint:raise ValueError('Parent/Fit normalizer mismatch')
    parent.load_state_dict(payload['model_state_dict'],strict=True)
    updated,record=refine(parent,tensors,selection,args.seed)
    args.output.mkdir(parents=True)
    record['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    torch.save({'model_state_dict':updated.state_dict(),**record},args.output/'selected.pt')
    rebuilt=FeedbackModel(parent,record['context']['mode'],record['context']['dimension'],target_dim=fit['target'].shape[1]).eval()
    rebuilt.load_state_dict(torch.load(args.output/'selected.pt',map_location='cpu',weights_only=False)['model_state_dict'],strict=True)
    a=predict_raw(updated,tensors,selection['pre'][:33],selection['condition'][:33])
    b=predict_raw(rebuilt,tensors,selection['pre'][:33],selection['condition'][:33])
    if not np.array_equal(a,b):raise ValueError('Native reload mismatch')
    normalizer={name:getattr(tensors.standardizer,name).numpy() for name in ['pre_mean','pre_scale','condition_mean','condition_scale','target_mean','target_scale']}
    np.savez_compressed(args.output/'normalizer.npz',**normalizer)
    record['reload_exact']=True
    (args.output/'RESULT.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'selected_candidate':record['selected_candidate'],'reload_exact':True}))

if __name__=='__main__':main()
