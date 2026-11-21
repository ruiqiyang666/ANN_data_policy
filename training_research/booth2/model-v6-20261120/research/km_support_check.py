"""Explicitly expose the empirical KM's unsupported-age case."""
import numpy as np
import pandas as pd
import train as T

model=dict(family='km_age',times=[2.,4.],survival=[.5,0.],tail=28.)
c,q,details=T.predict_model(model,pd.DataFrame(dict(age_h=[5.])))
# The declared fallback is a point mass at support+24h-age=23h, not immediate failure.
np.testing.assert_allclose(q,[[23,23,23]])
np.testing.assert_array_equal(c[0],(T.H>=23).astype(float))
assert details['unsupported_age_fraction']==1
print('Unsupported KM age fallback: PASS')
