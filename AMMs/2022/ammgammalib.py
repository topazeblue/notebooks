"""
Functions for analyzing AMM Gamma

:AMMSim:          AMM simulator class
:gamma_gain:      the gain of an arbitrageur due to their "Gamma" against the AMM
:fee_payment:     how much fees to be paid on a transaction

TECHNICAL HELPER FUNCTIONS
:create_prices:   creates a series of prices
:apply:           equivalent to lambda p,rg: np.array([func(x) for x in rg])
:A:               alias for Apply

LOCATION & COPYRIGHT

:copyright:     (c) Copyright Stefan LOESCH / topaze.blue 2022
:license:       MIT
:canonicurl:    https://github.com/topazeblue/notebooks/blob/main/_lib/AMMGammaLib.py
:display:       print("AMMGammaLib version {0.__VERSION__} ({0.__DATE__})".format(AMMSim))
"""
__VERSION__ = "1.2"
__DATE__ = "26/Dec/2022"

import numpy as _np
from collections import namedtuple as nt
from math import log as _log, exp as _exp, sqrt as _sqrt


def gamma_gain(p0, p1, N=1, feepc=0):
    """
    the gain of an arbitrageur due to their "Gamma" against the AMM 
    
    :p0:         price before move
    :p1:         price after move
    :N:          notional
    :feepc:      percentage fee (0.01=1%)
    :returns:    p1-sqrt(p0*p1) if p1 > p0, else sqrt(p0*p1)-p0

    NOTE: the arbitrageur "Gamma" is due to the fact that a constant product AMM 
    always trades at the geometric average of its current and new marginal price.
    If we assume that marginal prices correspond to market prices then after every
    market move p0->p1 the AMM allows trading a sqrt(p0*p1) which is (1) always 
    better for the arbitrageurs, and (2) same on the way up and down
    """
    if feepc == 0:
        return N*abs(p1-_sqrt(p0*p1))
    else:
        if p0<p1:
            return max(p1 - _sqrt(p0*(1+feepc)*p1), 0)
        else:
            return max(_sqrt(p0*(1-feepc)*p1) - p1, 0)

            
    #return N*(p1-_sqrt(p0*p1)) if p1 > p0 else N*(_sqrt(p0*p1)-p1)
    
def fee_payment(fee, p1, N=1):
    """
    how much fees to be paid on a transaction
    
    :fee:        percentage fee
    :p1:         price after move
    :N:          notional
    :returns:    fee*N*p1
    """
    return fee*N*p1



trade = nt("trade", "dx, dy, p0, p_trade, p1, bleed, fee, reverted")
class AMMSim:
    """
    AMM Simulator class
    
    :p0:      initial price (in y per x)
    :tvl0:    the initial tvl (measured in y)
    :feepc:   default percentage fees
    
    PROPERTIES SET BY THE CONSTRUCTOR
    :x:         current amount of risk asset held
    :k:         pool constant
    :bleed:     cumulative bleed = (market price - trade price) * trade volume [cost to AMM]
    :fees:      cumulative fees [income to AMM]
    :ntrades:   number of trades (including reverted)
    :nreverted: number of reverted trades
    """
    __VERSION__ = __VERSION__
    __DATE__ = __DATE__
    
    def __init__(self, p0=100, tvl0=100, feepc=0):
        y0 = tvl0/2
        x0 = y0/p0
        self.k = x0*y0
        self.x = x0
        self.feepc = feepc
        self.bleed = 0      
        self.fees = 0     
        self.ntrades = 0
        self.nreverted = 0

    def copy(self, p0=None, tvl0=None, feepc=None):
        """returns a copy of the object"""
        if p0 is None: p0 = self.p_marg
        if tvl0 is None: tvl0 = self.tvl
        if feepc is None: feepc = self.feepc
        return self.__class__(p0, tvl0, feepc)

    def aggregate(self, simlist):
        """
        aggregates a list of sims into a copy of the master sim*

        :simlist:       an iterable of sims that have been run
        :returns:       a copy of the current object, augmented with the aggregate
                        information from simlist PROVIDED that there have been no
                        trades recorded on the current sim
        """
        if self.ntrades:
            raise ValueError("You must only use aggregate on an unused sim (ntrades={self.ntrades})")
        aggr = (_np.array((o.bleed, o.fees, o.ntrades, o.nreverted)) for o in simlist)
        aggr = sum(aggr)
        newobj = self.copy()
        newobj.bleed        = aggr[0]
        newobj.fees         = aggr[1]
        newobj.ntrades      = aggr[2]
        newobj.nreverted    = aggr[3]
        return newobj
        
    def __call__(seld, *args, **kwargs):
        """alias for copy"""
        return self.copy(*args, **kwargs)
        
        
    @property
    def y(self):
        """current amount of numeraire asset y"""
        return self.k / self.x
    
    @property
    def tvl(self):
        """total value locked (in y)"""
        return self.y*2
    
    @property
    def p_marg(self):
        """current marginal price (in y per x)"""
        return self.y / self.x
    
    def trade_to(self, price, feepc=None):
        """
        trades to a new price
        
        :price:     the new price (in y per x)
        :feepc:     percentage trade fee (if None, use instance defaults)
        :returns:   trade namedtuple
                    :dx:            change in risk asset (negative = AMM sells)
                    :dy:            ditto numeraire asset
                    :p0:            price before trade (in y per x)
                    :p_trade:       effective price of the trade (ditto)
                    :p1:            price after trade (ditto)
                    :bleed:         bleed of the trade (arbitrageur income)
                    :fee:           fee of the trade (arbitrgeur expense)
                    :reverted:      if True, trade has not been counted
        
        Note: the counter self.bleed is increased by the amount of bleed suffered
        """
        if self.p_marg == price:
            return trade(0, 0, price, price, price, 0, 0, True)
        
        if feepc is None: feepc = self.feepc
        
        x0 = self.x
        y0 = self.y
        p0 = self.p_marg
        self.x = _sqrt(self.k/price)
        dx = self.x - x0
        dy = self.y - y0
        p_eff0 = -dy/dx
        p_eff = -(dy*(1-feepc))/dx
        p1 = self.p_marg
        bleed = -dx*(p1-p_eff0)
        fee = abs(dy)*feepc
        self.ntrades += 1
        if fee>bleed:
            self.x = x0
            self.nreverted += 1
        else:
            self.bleed += bleed
            self.fees += fee
        return trade(dx, dy, p0, p_eff, p1, bleed, fee, fee>bleed)
    
    @property
    def npassed(self):
        """number of trades passed"""
        return self.ntrades-self.nreverted

    @property
    def pcreverted(self):
        """percentage reverted"""
        try:
            return self.nreverted/self.ntrades
        except ZeroDivisionError:
            return None

    @property
    def pcpassed(self):
        """percentage passed"""
        try:
            return self.npassed/self.ntrades
        except ZeroDivisionError:
            return None

    @property
    def ammvalcapturepc(self):
        """fees as percentage of bleed (value captured by AMM LPs)"""
        try:
            return self.fees/self.bleed
        except ZeroDivisionError:
            return None  

    @property
    def arbvalcapturepc(self):
        """1 - fees as percentage of bleed (value captured by arbitrageurs)"""
        return 1-self.ammvalcapturepc

    @property
    def totalvalcapture(self):
        """total value captured (=bleed)"""
        return self.bleed

    @property
    def ammvalcapture(self):
        """value captured by AMM LPs (ie fees)"""
        return self.fees
    
    @property
    def arbvalcapture(self):
        """value captured by arbitrageurs (ie total - fees)"""
        return self.totalvalcapture-self.ammvalcapture




def create_prices(sig, N=10, p0=100, T=1, add_p0=False):
    """
    creates a series of prices
    
    :sig:    lognormal volatility (0.1 = 10%)
    :N:      number of steps
    :p0:     price at t=0
    :T:      total time period
    :add_p0: if True, the first price is p0
    """
    dt = T/N
    vol_sqrt_dt = sig*_sqrt(dt)
    exp_fctr = _exp(-sig*sig*dt)
    #print(f"sig={sig}, dt={dt}, vol_sqrt_dt={vol_sqrt_dt}, exp_fctr={exp_fctr}")
    random_changes = (vol_sqrt_dt*_np.random.randn()*exp_fctr for _ in range(N))
    marg_multipliers = (1+x for x in random_changes)
    multipliers = _np.cumprod(tuple(marg_multipliers))
    prices = (p0 * x for x in multipliers)
    prices = _np.array(tuple(prices))
    if add_p0:
        prices = _np.concatenate(([p0], prices))
    return prices   

def apply(func, rg):
    """
    applies `func` to `rg` and returns result as np.array
    
    equivalent to `lambda f,r: np.array([f(x) for x in r])`
    """
    return _np.array([func(x) for x in rg])
A=apply