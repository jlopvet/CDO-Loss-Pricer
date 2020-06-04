import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.stats import norm
from scipy.stats import uniform
import scipy.integrate as integrate
import scipy.special as special
from datetime import datetime
from numba import jit


def parser(mat):
    if "M" in mat:
        return float(mat[:-1])/12
    elif "D" in mat:
        return float(mat[:-1])/360
    elif "Y" in mat:
        return float(mat[:-1])
    elif "W" in mat:
        return float(mat[:-1])/360*7


class CreditCurve():
    def __init__(self, tenors, spreads, R):
        self.R = R
        h = [e / 100 / 100 /(1 - R) for e in spreads]
        self.h_ = h
        self.h = interpolate.interp1d(tenors, h, kind="previous", fill_value=(h[0], h[-1]), bounds_error=False)
        self.tenors = tenors
        self.P = [1 - np.exp(-integrate.quad(lambda t: self.h(t), 0, t)[0]) for t in self.tenors]

        self.Pr = interpolate.interp1d(self.tenors, self.P, kind="cubic", fill_value=(self.P[0], self.P[-1]), bounds_error=False)

    def p(self, t):
        return self.Pr(t)

    def show(self):
        xnew = np.arange(0, 15, 0.05)
        ynew = self.Pr(xnew)
        plt.plot(self.tenors, self.P, 'o', xnew, ynew, '-')
        plt.show()


class DiscountCurve():
    def __init__(self, tenors, yields):

        self.r = interpolate.interp1d(tenors, yields, kind="cubic", fill_value=(yields[0], yields[-1]), bounds_error=False)

    def rate(self, t):
        return self.r(t)

    def P(self, t):
        return np.exp(-self.rate(t) * t)

    def f(self, t):
        return - ( np.log(self.P(t+0.0001)) - np.log(self.P(t)) ) / 0.0001

    def show(self):
        xnew = np.arange(0, 15, 0.1)
        ynew = self.r(xnew)
        plt.plot(xnew, ynew, '-')
        plt.show()

    def show_f(self, T):
        xnew = np.arange(0, T, 0.1)
        ynew = [self.f(t) for t in xnew]
        plt.plot(xnew, ynew, '-')
        plt.show()

class CDO():
    def __init__(self, valuationDate, maturity, creditCurves, discountCurve, K1, corr1, K2, corr2, Nominal=10000000):

        self.creditCurves = creditCurves
        self.discountCurve = discountCurve
        self.T = (maturity - valuationDate).days / 360
        self.R = np.array([cc.R for cc in creditCurves])
        self.K1 = K1
        self.corr1 = corr1
        self.K2 = K2
        self.corr2 = corr2
        self.Nominal = Nominal

        if K2 < K1:
            raise ValueError("K2 < K1 !")

        self.calculateMeanLoss()

    def calculateMeanLoss(self, Nt=20, Nv=200, Nk=1000):

        Nn = len(creditCurves)

        # Etape 1 on crée les grilles qui permettent de discrétiser k, v, t, n
        K = np.linspace(0, 1, Nk, endpoint=True)
        V = np.linspace(-12, 12, Nv, endpoint=True)
        t = np.linspace(0, self.T, Nt, endpoint=True)

        cp1 = self.__init_cond_prob(self.corr1, t, V)
        cp2 = self.__init_cond_prob(self.corr2, t, V)

        # Pour accélérer le code on précalcule la densité gaussienne
        norm_pdf = np.zeros(Nv)
        for jv, v in enumerate(V):
            norm_pdf[jv] = norm.pdf(v)

        L1 = hw_losses_nh(cp1, self.R, self.corr1, self.K1, Nn, t, K, V, norm_pdf)
        self.Loss1 = interpolate.interp1d(t, L1, kind="quadratic", fill_value=(L1[0], L1[-1]), bounds_error=False)

        L2 = hw_losses_nh(cp2, self.R, self.corr2, self.K2, Nn, t, K, V, norm_pdf)
        self.Loss2 = interpolate.interp1d(t, L2, kind="quadratic", fill_value=(L2[0], L2[-1]), bounds_error=False)

        t_ = np.arange(0, self.T, 0.001)
        plt.plot(t, L1, 'o', t_, self.Loss1(t_), '-')
        plt.plot(t, L2, 'o', t_, self.Loss2(t_), '-')
        plt.title("L(t)")
        plt.show()

    def default_leg(self):
        t_ = np.arange(0, self.T, 0.001)
        DL1 = 0
        DL2 = 0

        for jt in range(len(t_)-1):
            DL1 += self.discountCurve.P((t_[jt+1]+t_[jt])/2) * (self.Loss1(t_[jt+1]) - self.Loss1(t_[jt]))
            DL2 += self.discountCurve.P((t_[jt+1]+t_[jt])/2) * (self.Loss2(t_[jt+1]) - self.Loss2(t_[jt]))

        if self.K1 == 0:
            return DL2 * self.Nominal / self.K2
        else:
            return (DL2 - DL1) * self.Nominal / (self.K2 - self.K1)


    def __init_cond_prob(self, rho, T, V):
        Nn = len(self.creditCurves)
        Nt = len(T)
        Nv = len(V)

        # Calculate conditional probabilities
        cp = np.zeros((Nn, Nt, Nv))

        for jn, cc in enumerate(creditCurves):
            for jt, t in enumerate(T):
                for jv, v in enumerate(V):
                    cp[jn, jt, jv] = norm.cdf( (norm.ppf(cc.p(t)) - rho*v) / np.sqrt(1-rho**2) )
        return cp



@jit
def hw_losses_nh(cp, R, rho, B, Nn, T, K, V, norm_pdf):

    Nk = len(K)
    Nt = len(T)
    Nv = len(V)

    pi = np.zeros((Nk, Nv))
    A = np.zeros((Nk, Nv))
    add_cond = np.zeros(Nk)
    add_mean = np.zeros(Nk)
    losses = np.zeros((Nt, Nk))

    L = np.zeros(Nt)
    p_default = 0
    sum_ = 0

    for jt in range(Nt):
        for jv in range(Nv):
            pi[0][jv] = 1.
            A[0][jv] = 0.

        for jk in range(1, Nk):
            for jv in range(Nv):
                pi[jk][jv] = 0.
                A[jk][jv] = 0.

        for jn in range(Nn):
            L_j = 1/Nn * (1 - R[jn])

            for jv in range(Nv):
                p_default = cp[jn][jt][jv]
                for jk in range(Nk):
                    add_cond[jk] = 0.
                    add_mean[jk] = 0.

                for jk in range(Nk):
                    A_kpL_j = A[jk][jv] + L_j
                    ujk = jk
                    while ((ujk + 1 < Nk) and (A_kpL_j >= K[ujk + 1])):
                        ujk += 1

                    if ujk > jk:
                        add_cond[jk] -= pi[jk][jv] * p_default
                        add_cond[ujk] += pi[jk][jv] * p_default
                        if (pi[ujk][jv] + pi[jk][jv] * p_default == 0):
                            add_mean[ujk] = 0
                        else:
                            add_mean[ujk] += (pi[jk][jv] * p_default * (A_kpL_j - A[ujk][jv])) / (pi[ujk][jv] + pi[jk][jv] * p_default)
                    else:
                        add_mean[jk] += p_default * L_j

                for jk in range(Nk):
                    pi[jk][jv] += add_cond[jk]
                    A[jk][jv] += add_mean[jk]

        dv = V[1] - V[0]
        for jk in range(Nk):
            losses[jt][jk] = 0
            for jv in range(Nv):
                losses[jt][jk] += pi[jk][jv] * norm_pdf[jv] * dv

    ML    = np.zeros(Nt)
    
    for jt in range(Nt):
        ML[jt] = 0.

        for jk, k in enumerate(K):
            ML[jt] += min(k, B) * losses[jt][jk]


    return ML


if __name__ == "__main__":

    valuationDate = datetime(2019, 12, 30)
    maturity = datetime(2024, 12, 24)
    R = 0.4
    K1 = 0.03
    corr1 = 0.4
    K2 = 0.06
    corr2 = 0.6
    Nominal = 10000000 # 10 0000 000


    # Create discount curve
    tenors = []
    yields = []
    df = pd.read_excel("s261.xlsx")
    for (idx, row) in df.iterrows():
        yields.append(row[df.columns[3]]/100)
        tenors.append(parser(row[df.columns[0]]))
    discountCurve = DiscountCurve(tenors, yields)

    # Create credit curves
    tenors_ = ["6 Mo", "1 Yr", "2 Yr", "3 Yr", "4 Yr", "5 Yr", "7 Yr", "10 Yr"]
    tenors = [0.5, 1, 2, 3, 4, 5, 7, 10]
    creditCurves = []
    df = pd.read_excel("spreads.xlsx")
    for (idx, row) in df.iterrows():
        spreads = []
        for t in tenors_:
            spreads.append(row[t])
        creditCurves.append( CreditCurve(tenors, spreads, R) )


    cdo = CDO(valuationDate, maturity, creditCurves, discountCurve, K1, corr1, K2, corr2)
    print(cdo.default_leg())