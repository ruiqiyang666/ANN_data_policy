import sys
sys.dont_write_bytecode=True
import torch
from risk_models import distribution
from common_v2 import EVAL,dump

torch.manual_seed(42)
logits=torch.cat([torch.randn(10000,10)*8,torch.full((2,10),-1000.),torch.full((2,10),1000.)])
p,q=distribution(logits)
assert ((p>=0)&(p<=1)).all() and torch.isfinite(p).all()
torch.testing.assert_close(q.sum(1),torch.ones(len(q)),atol=2e-6,rtol=0)
assert (q>=0).all()
dump({'passed':True,'cases':len(p),'max_probability':float(p.max()),'min_probability':float(p.min()),
      'regression':'float32 event masses must not produce probability above1; conditional time remains normalized under underflow'},EVAL/'probability_regression.json')
print('10004 probability boundary and conditional normalization cases passed.')
