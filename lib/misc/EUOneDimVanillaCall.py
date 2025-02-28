import numpy as np
from scipy.stats import norm
import tensorflow as tf
tf.keras.backend.set_floatx(tf.keras.backend.floatx())
import tensorflow_probability as tfp

class EUVanillaCallPut(object):
    def __init__(self, call_or_put):
        if call_or_put.lower() == 'call':
            self.call_or_put = 'call'
        elif call_or_put.lower() == 'put':
            self.call_or_put = 'put'
        else:
            raise ValueError()

    @staticmethod
    def d1(S, K, r, sig, tau, q=0):
        return (np.log(S / K) + (r - q + sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))

    @staticmethod
    def d2(S, K, r, sig, tau, q=0):
        d1 = (np.log(S / K) + (r - q + sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))
        return d1 - sig * np.sqrt(tau)

    @staticmethod
    def price_call(S, K, r, sig, tau, q=0):
        d1 = (np.log(S / K) + (r - q + sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))
        d2 = d1 - sig * np.sqrt(tau)
        return np.exp(-r * tau) * (np.exp((r - q) * tau) * S * norm.cdf(d1) - K * norm.cdf(d2))

    @staticmethod
    def delta_call(S, K, r, sig, tau, q=0):
        d1 = (np.log(S / K) + (r - q + sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))
        return np.exp(-q * tau) * norm.cdf(d1)

    @staticmethod
    def gamma_call(S, K, r, sig, tau, q=0):
        d1 = (np.log(S / K) + (r - q + sig ** 2 / 2) * tau) / (sig * np.sqrt(tau))
        return np.exp(-q * tau) * norm.pdf(d1) / (S * sig * np.sqrt(tau))

    def price(self, S, K, r, sig, tau, q=0):
        call_price = self.price_call(S, K, r, sig, tau, q)

        if self.call_or_put == 'call':
            return call_price
        else:
            # # # put call parity - with continuous dividends
            put_price = call_price - S * np.exp(-q * tau) + K * np.exp(-r * tau)
            # if q != 0:
            #     raise NotImplementedError
            # put_price = call_price - S + K * np.exp(-r * tau)
            return put_price

    def delta(self, S, K, r, sig, tau, q=0):
        call_delta = self.delta_call(S, K, r, sig, tau, q)
        if self.call_or_put == 'call':
            return call_delta
        else:
            # # # put call parity - with continuous dividends
            put_delta = call_delta - np.exp(-q * tau)
            # # archived: no dividends
            # # if q != 0:
            # #     raise NotImplementedError
            # # put_delta = call_delta - 1
            return put_delta

    def gamma(self, S, K, r, sig, tau, q=0):
        call_gamma = self.gamma_call(S, K, r, sig, tau, q)
        if self.call_or_put == 'call':
            return call_gamma
        else:
            # # # put call parity
            return call_gamma

class EUExchangeCall(object):
    """
    should work appropriately
    """
    def __init__(self):
        return None

    @staticmethod
    def d1_jk(S_k, S_j, K_jk, sigma_k, sigma_j, rho_jk, t, T, q=0):
        if q != 0:
            raise NotImplementedError("Not sure how this looks with dividend rates")
        tau = T - t  # time to maturity
        sigma_jk = np.sqrt(sigma_j ** 2 + sigma_k ** 2 - 2 * rho_jk * sigma_j * sigma_k)

        return np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) + 0.5 * sigma_jk * np.sqrt(tau)

    @staticmethod
    def d2_jk(S_k, S_j, K_jk, sigma_k, sigma_j, rho_jk, t, T, q=0):
        if q != 0:
            raise NotImplementedError("Not sure how this looks with dividend rates")
        tau = T - t  # time to maturity
        sigma_jk = np.sqrt(sigma_j ** 2 + sigma_k ** 2 - 2 * rho_jk * sigma_j * sigma_k)

        return np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) - 0.5 * sigma_jk * np.sqrt(tau)

    @staticmethod
    def price_call(S_k, S_j, K_jk, sigma_k, sigma_j, rho_jk, t, T, q=0):
        if q != 0:
            raise NotImplementedError("Not sure how this looks with dividend rates")
        tau = T - t  # time to maturity
        sigma_jk = np.sqrt(sigma_j ** 2 + sigma_k ** 2 - 2 * rho_jk * sigma_j * sigma_k)

        d1_jk = np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) + 0.5 * sigma_jk * np.sqrt(tau)
        d2_jk = np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) - 0.5 * sigma_jk * np.sqrt(tau)

        return norm.cdf(d1_jk) * S_k - K_jk * S_j * norm.cdf(d2_jk)

    @staticmethod
    def delta_call(S_k, S_j, K_jk, sigma_k, sigma_j, rho_jk, t, T, q=0):
        if q != 0:
            raise NotImplementedError("Not sure how this looks with dividend rates")

        tau = T - t  # time to maturity
        sigma_jk = np.sqrt(sigma_j ** 2 + sigma_k ** 2 - 2 * rho_jk * sigma_j * sigma_k)

        d1_jk = np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) + 0.5 * sigma_jk * np.sqrt(tau)
        d2_jk = np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) - 0.5 * sigma_jk * np.sqrt(tau)

        dj_C_jk = -K_jk * norm.cdf(d2_jk) # should be either M or M x 1
        dk_C_jk = norm.cdf(d1_jk)

        M = S_k.shape[0]

        return dj_C_jk, dk_C_jk
        return np.concatenate([dj_C_jk.reshape((M, 1)), dk_C_jk.reshape((M, 1))], axis=-1)  # M x 2




    @staticmethod
    def gamma_call(S_k, S_j, K_jk, sigma_k, sigma_j, rho_jk, t, T, q=0):
        if q != 0:
            raise NotImplementedError("Not sure how this looks with dividend rates")

        tau = T - t  # time to maturity
        sigma_jk = np.sqrt(sigma_j ** 2 + sigma_k ** 2 - 2 * rho_jk * sigma_j * sigma_k)

        d1_jk = np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) + 0.5 * sigma_jk * np.sqrt(tau)
        d2_jk = np.log(S_k / (K_jk * S_j)) / (sigma_jk * np.sqrt(tau)) - 0.5 * sigma_jk * np.sqrt(tau)

        # # # they should all be either M or M x 1
        djj_C_jk = S_k / (S_j ** 2) / (np.sqrt(2 * np.pi) * sigma_jk * np.sqrt(tau)) * np.exp(-d1_jk ** 2 / 2)
        djk_C_jk = S_j ** (-1) / (-np.sqrt(2 * np.pi) * sigma_jk * np.sqrt(tau)) * np.exp(-d1_jk ** 2 / 2)
        dkj_C_jk = K_jk / S_k / (-np.sqrt(2 * np.pi) * sigma_jk * np.sqrt(tau)) * np.exp(-d2_jk ** 2 / 2)
        dkk_C_jk = S_k ** (-1) / (np.sqrt(2 * np.pi) * sigma_jk * np.sqrt(tau)) * np.exp(-d1_jk ** 2 / 2)

        M = S_k.shape[0]

        hess_C_jk = np.zeros([M, 2, 2])
        hess_C_jk[:, 0, 0] = djj_C_jk
        hess_C_jk[:, 0, 1] = dkj_C_jk
        hess_C_jk[:, 1, 0] = djk_C_jk
        hess_C_jk[:, 1, 1] = dkk_C_jk

        return djj_C_jk, dkj_C_jk, djk_C_jk, dkk_C_jk
        return hess_C_jk

