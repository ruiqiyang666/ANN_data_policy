"""Focused mathematical guards, including the below-censor quantile branch."""
import numpy as np
import train as T

taus=np.array([[.1,.5,.9]])
weights=T.cqr_weights(taus,np.array([[.5]]))
np.testing.assert_allclose(weights,[[1,0,.8]])
# With censor bound 5 and a q=.1 prediction below it, the gradient must push
# towards the censor bound, not apportion all mass to the far-away pseudo point.
assert weights[0,0]==1
c,q=T.mixture_outputs([(np.array([[np.log(4),np.log(4)]]),np.array([.5,.5]),np.array([[.25,.75]]))])
np.testing.assert_allclose(q[0,1],4,rtol=1e-6)
assert np.all(np.diff(c,axis=1)>=0) and np.all(np.diff(q,axis=1)>0)
c,q,u=T.surv_outputs(np.array([1.,3.]),np.array([[.8,.4]]),27)
np.testing.assert_allclose(q,[[1,3,27]])
assert u.tolist()==[[False,False,True]]
print('CQR weights, mixture median, survival tail: PASS')
