"""
Simple BlackScholes option pricing object

:copyright:     (c) Copyright Stefan LOESCH / topaze.blue 2022; ALL RIGHTS RESERVED
:license:       MIT
:canonicurl:    https://github.com/topazeblue/notebooks/blob/main/_lib/BlackScholesLib.py

"""
__VERSION__ = "1.0"
__DATE__ = "22/Dec/2022"

from math import sqrt, log, exp, pi
from scipy.stats import norm
from copy import copy

class CallPutForward():
    """
    calls, puts and other options with the same gamma
    
    :K:         strike
    :S:         spot
    :T:         maturity
    :sig:       volatility
    :t:         current time
    :r:         interest rate
    :rf:        foreign interest rate
    :N:         notional
    :ND:        additional delta notional (Forward @ K)
    """
    __VERSION__ = __VERSION__
    __DATE__ = __DATE__
    
    def __init__(self, sig, K=None, S=None, T=None, r=None, rf=None, t=None, N=None, ND=None):
        self.sig = sig
        self.K   = K if K else 100
        self.S   = S if S else 100
        self.T   = T if T else 1
        self.t   = t if t else 0
        self.rf  = rf if rf else 0
        self.r   = r  if r  else 0
        self.N   = N  if not N is None else 1
        self.ND  = ND  if ND  else 0
        
    _VARIABLES = set("K,S,T,sig,t,rf,r,N,ND,F,df,dff".split(","))
    _SPECIAL = set("F,df,dff".split(","))
    
    def setv(self, **kvdict):
        """sets value and returns new object"""
        variables = set(kvdict.keys())
        wrongvars = variables-self._VARIABLES
        #print("[setv] VARIABLES", self._VARIABLES)
        #print("[setv] KEYS", keys)
        #print("[setv] WRONGKEYS", wrongvars)
        if wrongvars:
            raise ValueError(f"Unknown keys ({wrongvars})", variables, self._VARIABLES)
        specialvars = variables.intersection(self._SPECIAL)
        regvars = variables - specialvars
        newobj = copy(self)
        for k in regvars:
            newobj.__setattr__(k,kvdict[k])
        if "F" in specialvars:
            newobj.S = kvdict["F"]/newobj.ff
        if "df" in specialvars:
            newobj.r = -log(kvdict["df"])/newobj.ttm
        if "dff" in specialvars:
            newobj.r = -log(kvdict["dff"])/newobj.ttm
        return newobj
        
    @property
    def F(self):
        """forward"""
        return self.S * self.ff
    
    @property
    def df(self):
        """discount factor"""
        return exp(-self.r*(self.T-self.t))
    
    @property
    def dff(self):
        """foreign discount factor"""
        return exp(-self.rf*(self.T-self.t))
    
    @property
    def ff(self):
        """forward factor = F/S"""
        return self.df / self.dff
    
    @property
    def ttm(self):
        """time to maturity"""
        return self.T-self.t
        
    @property
    def d1(self):
        """d1"""
        return self.dd[0]
    
    @property
    def d2(self):
        """d2"""
        return self.dd[1]
    
    @property
    def Nd1(self):
        """N(d1)"""
        return norm.cdf(self.d1)
    
    @property
    def Nd2(self):
        """N(d2)"""
        return norm.cdf(self.d2)    
    
    @property
    def dd(self):
        """(d1, d2)"""
        ttm = self.ttm
        sig_sqrt_t = self.sig * sqrt(ttm)
        
        d1 = (log(self.S/self.K) + (self.r - self.rf + 0.5*self.sig**2)*(ttm)) / sig_sqrt_t
        d2 = d1 - sig_sqrt_t
        return d1, d2
    
    @property
    def pv(self):
        """present value"""
        #return (self.S-100)**3
        d1, d2 = self.dd
        Nd1 = norm.cdf(d1)
        Nd2 = norm.cdf(d2)
        option = self.N*(self.F * Nd1 - self.K * Nd2)
        delta = self.ND*(self.F-self.K)
        return self.df * (option + delta)
    
    EPS = 0.0001
    @property
    def delta(self):
        """risk asset delta, dPV/dS"""
        dS = self.S * self.EPS
        pv1 = self.setv(S=self.S+dS).pv
        pv2 = self.setv(S=self.S-dS).pv
        return (pv1-pv2)/(2*dS)
    
    @property
    def cashdelta(self):
        """cash delta = S*delta"""
        return self.S * self.delta
    
    @property
    def gamma(self):
        """gamma"""
        dS = self.S * self.EPS
        pv1 = self.setv(S=self.S+dS).pv
        pv2 = self.setv(S=self.S-dS).pv
        pv3 = self.pv
        return (pv1+pv2-2*pv3)/(dS**2)
    
    @property
    def cashgamma(self):
        """cash gamma = S^2 * gamma"""   
        return self.S**2 * self.gamma
    
    @property
    def riskgamma(self):
        """risk gamma = S * gamma"""   
        return self.S * self.gamma
    
    @property
    def pv_atm(self):
        """at the money approximate PV"""
        return 1/sqrt(2*pi)*self.S*self.sig*sqrt(self.ttm)
        
        
