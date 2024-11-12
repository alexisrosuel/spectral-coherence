# spectral-coherence
Lightweight package to compute estimates of the spectral coherence matrix of time series. Uses apple MLX package.

Here is the most basic usage: 
```python
import numpy as np
from spectral_coherence import smoothed_periodograms, coherences 

# generate M independent AR(1) processes with different parameters
n_samples, n_features = 1000, 4
thetas = np.linspace(-0.7, 0.7, n_features)
epsilon = np.random.normal(size=(n_samples, n_features))
y = np.zeros((n_samples, n_features))
y[0, :] = epsilon[0, :]
for t in range(1, n_samples):
    y[t, :] = thetas * y[t-1, :] + epsilon[t, :]

# cast to mx.array, the package only accepts mx.array as input
y = mx.array(y)

# estimate the spectral density and the coherence from the sample 
S_hats, freqs = smoothed_periodograms(y, B=31)
C_hats, freqs = coherences(y, B=31)
```
