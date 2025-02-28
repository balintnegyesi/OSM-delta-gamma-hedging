import numpy as np
from scipy.integrate import odeint

class HJBODESolver(object):
    def __init__(self, num_time_interval, dim, total_time, P_T=None, Q_T=None, R_T=None):
        self.num_time_interval = num_time_interval
        self.dim = dim
        self.total_time = total_time

        if P_T is None:
            self.P_T = np.eye(self.dim)
        else:
            self.P_T = P_T

        if Q_T is None:
            self.Q_T = np.zeros(self.dim)
        else:
            self.Q_T = Q_T

        if R_T is None:
            self.R_T = 0
        else:
            self.R_T = R_T

        self.delta_t = self.total_time / self.num_time_interval
        self.t = np.arange(0, self.num_time_interval + 1) * self.delta_t
        self.t = np.linspace(0, self.total_time, self.num_time_interval + 1)
        self.t_reverse = np.linspace(self.total_time, 0, self.num_time_interval + 1)

    def dy_dt(self, y, t, d):
        """
        implements the right hand side of a system of ODEs:
        dy_dt = f(y)
        """
        retval_len = d ** 2 + d + 1  # expanding the matrix valued Ricatti to vectors
        retval = np.zeros(shape=retval_len)

        # first d^2 equations
        for i in range(d ** 2):
            # # # the matrix value part
            # conversion of i to tuple pj, which corresponds to element pj of the mtx
            # p=0...d-1, j=0...d-1, i=0...d
            fraction, intpart = np.modf(i / d)
            p = int(intpart)
            j = i - p * d  # should be integer as well

            for k in range(d):
                P_plus_PT_pk = y[p * d + k] + y[k * d + p]
                P_plus_PT_kj = y[k * d + j] + y[j * d + k]

                retval[i] += P_plus_PT_pk * P_plus_PT_kj

        # second d equations
        for idx in range(d ** 2, d ** 2 + d):
            i = idx - d ** 2
            for k in range(d):
                P_plus_PT_ik = y[i * d + k] + y[k * d + i]
                Q_k = y[d ** 2 + k]
                retval[idx] += 2 * P_plus_PT_ik * Q_k

        # last 1 equation
        for k in range(d):
            retval[-1] += y[d ** 2 + k] ** 2 - 2 * y[k * d + k]

        return retval

    def solve_odes(self):
        vector_solution = self.integrate_ode()
        P_t, Q_t, R_t = self.vector_to_matrices(vector_solution)

        self.P_t = P_t
        self.Q_t = Q_t
        self.R_t = R_t

        return P_t, Q_t, R_t

    def integrate_ode(self):
        # # # initial conditions
        # P_T = np.eye(self.dim)
        P_T = self.P_T
        # Q_T = np.zeros(self.dim)
        Q_T = self.Q_T
        # R_T = 0
        R_T = self.R_T

        # # # vectorized initial conditions
        y_T = np.zeros(shape=(self.dim ** 2 + self.dim + 1))
        y_T[0: self.dim ** 2] = P_T.flatten(order='C')  # row-major order
        y_T[self.dim ** 2: self.dim ** 2 + self.dim] = Q_T
        y_T[-1] = R_T

        solution_reverse = odeint(self.dy_dt, y_T, self.t_reverse, args=(self.dim,))

        return solution_reverse

    def vector_to_matrices(self, solution):
        P_t = np.zeros(shape=[self.num_time_interval + 1, self.dim, self.dim])
        Q_t = np.zeros(shape=[self.num_time_interval + 1, self.dim])
        R_t = np.zeros(shape=[self.num_time_interval + 1, 1])

        for i_rev in range(len(self.t_reverse)):
            # # # python backward slicing: -1 last element --> -(i_rev+1) is the element to point to
            idx = -(i_rev + 1)
            P_t[idx, :, :] = np.reshape(solution[i_rev, 0: self.dim ** 2], newshape=(self.dim, self.dim), order='C')
            Q_t[idx, :] = solution[i_rev, self.dim ** 2: self.dim ** 2 + self.dim]
            R_t[idx, 0] = solution[i_rev, -1]

        return P_t, Q_t, R_t