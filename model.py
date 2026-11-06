"""
A/B Testing & Causal Inference Toolkit

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standard_normal_cdf
import math
import numpy as np

def standard_normal_cdf(z):
    # TODO: return P(Z <= z) for a standard normal Z, supporting float or numpy array input
    _sqrt2 = math.sqrt(2.0)
    if np.ndim(z) == 0:
        return 0.5 * math.erfc(-float(z) / _sqrt2)
    _erfc = np.vectorize(math.erfc, otypes=[float])
    z = np.asarray(z, dtype=float)
    return 0.5 * _erfc(-z / _sqrt2)

# Step 3 - pooled_proportion
def pooled_proportion(successes_a, total_a, successes_b, total_b):
    # TODO: Compute the pooled success proportion across two groups for the null of equal rates.
    return float((successes_a + successes_b) / (total_a + total_b))

