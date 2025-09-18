
import numpy as np

### Metrics ###

def compute_er(tx, rx):
    """Compute error rate and number of errors between two sequences"""
    assert tx.size == rx.size
    
    n_err = np.sum(tx != rx)
    er = n_err / tx.size
    return er, n_err