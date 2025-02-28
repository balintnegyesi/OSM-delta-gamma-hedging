import numpy as np
from scipy.stats import skew as scpskew


def get_pnl_statistics(pnl_sample):
    # # # shape: M x 1 x N + 1
    # # # returns: 1 x N + 1 shaped tensors corresponding to appropriate statistics
    retval = {'mean': np.mean(pnl_sample, axis=0), 'median': np.median(pnl_sample, axis=0),
              'variance': np.var(pnl_sample, axis=0), \
              'var01': np.quantile(pnl_sample, q=0.01, axis=0), 'var05': np.quantile(pnl_sample, q=0.05, axis=0), \
              'var95': np.quantile(pnl_sample, q=0.95, axis=0), 'var99': np.quantile(pnl_sample, q=0.99, axis=0), \
              'skew': scpskew(pnl_sample, axis=0)}

    retval['es01'] = np.zeros(retval['var01'].shape)
    retval['es05'] = np.zeros(retval['var05'].shape)
    retval['es95'] = np.zeros(retval['var95'].shape)
    retval['es99'] = np.zeros(retval['var99'].shape)
    retval['semivariance'] = np.zeros(retval['var01'].shape)

    for n in range(pnl_sample.shape[-1]):
        pnl_n = pnl_sample[..., n]
        retval['es01'][..., n] = np.mean(pnl_n[pnl_n <= retval['var01'][..., n]])
        retval['es05'][..., n] = np.mean(pnl_n[pnl_n <= retval['var05'][..., n]])
        retval['es95'][..., n] = np.mean(pnl_n[pnl_n >= retval['var95'][..., n]])
        retval['es99'][..., n] = np.mean(pnl_n[pnl_n >= retval['var99'][..., n]])
        retval['semivariance'][..., n] = np.var(pnl_n[pnl_n < retval['mean'][..., n]])


    return retval
