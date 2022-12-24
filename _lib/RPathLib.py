"""
Generating and processing random paths

:Path:      class representing a path
:PathGen:   path generator

LOCATION & COPYRIGHT

:copyright:     (c) Copyright Stefan LOESCH / topaze.blue 2022
:license:       MIT
:canonicurl:    https://github.com/topazeblue/notebooks
:display:       print("RPathLib version {0.__VERSION__} ({0.__DATE__})".format(RPath))
"""
__VERSION__ = "1.0"
__DATE__ = "23/Dec/2022"

import numpy as np
from math import sqrt, log, exp
from copy import copy

class RPath():
    """
    represents a single path that allows for subpath extractions
    
    :vals:      the values of the path
    :time:      the associated point time*
    :period:    default value for period
    :offset:    ditto for offset
    :bounds:    ditto for bounds
    :meta:      a dict of meta data associated to the path (info only; not uses here)
    
    NOTE. If you use PathGen.time to generate the time vector, each of them will be the 
    same np.array instance; this pattern is recommended: use the same time object for 
    different paths corresponding to the same sample, to ensure memory efficiency
    """
    __VERSION__ = __VERSION__
    __DATE__    = __DATE__
    
    def __init__(self, path, time, period=None, offset=None, bounds=None, meta=None):
        if len(path)!=len(time):
            raise ValueError("Lenght vals != length time", len(path), len(time))
        self._path = path
        self._time = time
        if meta is None: meta = dict()
        self.meta = meta
        if period is None: period = 1
        self._period = period
        self._offset = offset
        self._bounds = bounds

    def resample(self, period=None, offset=None, bounds=None):
        """
        returns a new path object with new default values of period, offset, bounds

        NOTE: __call__ is an alias for resample
        """
        newobj = copy(self)
        if not period is None: newobj._period=period
        if not offset is None: newobj._offset=offset
        if not bounds is None: newobj._bounds=bounds
        #print("[resample] {0._period}, {0._offset}, {0._bounds}".format(newobj))
        return newobj

    def __call__(self, *args, **kwargs):
        """alias for resample"""
        return self.resample(*args, **kwargs)

    @property
    def period(self):
        return self._period

    @property
    def offset(self):
        return self._offset

    @property
    def bounds(self):
        return self._bounds

        
    @staticmethod
    def extraction_pattern(N_points, period=None, offset=None, bounds=None):
        """
        creates the extraction pattern (eg 1,0,0,1,0,0,1,0,0,1)

        :N_points:  length of pattern vector (points, not periods)
        :period:    period length; eg period=3 gives 1,0,0,1,0,0,...
        :offset:    offset within period, eg period=4, offset=1 gives 0,1,0,0, 0,1,0,0, 0,1,0,0
        :bounds:    if True, includes the boundaries in the pattern
        """
        if period is None: period = 1
        if offset is None: offset = 0
        if bounds is None: bounds = True
        if period == 1:
            return np.array([1 for i in range(N_points)])

        if offset >= period:
            raise ValueError("Offset must be less than period", offset, period)
        result = np.array([1 if i % period == offset else 0 for i in range(N_points)])
        if bounds:
            result[0] = 1
            result[-1] = 1
        return result
    
    @staticmethod
    def apply_pattern(vec, pattern):
        """applies pattern to vec, ie only returns item for which it is unity"""
        return np.array([x for x,p in zip(vec, pattern) if p])
    
    def extract(self, vec, period=None, offset=None, bounds=None):
        """
        applies extraction_pattern to vec
        
        :period:    period length; eg period=3 gives 1,0,0,1,0,0,...
        :offset:    offset within period, eg period=4, offset=1 gives 0,1,0,0, 0,1,0,0, 0,1,0,0
        :bounds:    if True, includes the boundaries in the pattern
        """
        vec = np.array(vec)
        if period is None: period = self._period
        if offset is None: offset = self._offset
        if bounds is None: bounds = self._bounds
        pattern = self.extraction_pattern(len(vec), period, offset, bounds)
        return self.apply_pattern(vec, pattern)
    
    def path(self, period=None, offset=None, bounds=None):
        """
        extracts the path with using extract(path, period, offset, bounds)
        """
        return self.extract(self._path, period, offset, bounds)
        
    def time(self, period=None, offset=None, bounds=None):
        """
        extracts the time with using extract(path, period, offset, bounds)
        """
        return self.extract(self._time, period, offset, bounds)  
    
    def periodtime(self, period=None, offset=None, bounds=None):
        """
        extracts the time with using extract(path, period, offset, bounds)
        """
        if period is None: period = self._period
        return self.periodtime0 * period
    
    def N(self, period=None, offset=None, bounds=None):
        """
        extracts the time with using extract(path, period, offset, bounds); periods, not points
        """
        pattern = self.extraction_pattern(len(self._time), period, offset, bounds)
        return sum(pattern)-1

    @property
    def path0(self):
        """full path"""
        return self._path
    
    @property
    def time0(self):
        """full time"""
        return self._time

    @property
    def periodtime0(self):
        """base period (assumed constant, ie t[1]-t[0])"""
        return self._time[1]-self._time[0]
    
    @property
    def N0(self):
        """base length (number of periods, not points)"""
        return len(self._time)-1

    @property
    def T(self):
        """total time period"""
        return self._time[-1] - self._time[0]
    
class RPathGen():
    """
    creates random paths
    
    :method:      constants LOGNORM, NORM
    :val0:        initially value (DEFAULT_VAL0 if None)
    :T:           time period (DEFAULT_T if None)
    :N:           number of observation excluding val0 (DEFAULT_N if None)
    :params:      params as dict, DEFAULT_PARAMS[method] if None
    :kwargs:      alternative for params
    """
    LOGNORM = "lognorm"
    NORM = "norm"
    DEFAULT_PARAMS = {
        LOGNORM: {"sig": 0.10, "mu": 0},
        NORM:    {"sig": 1,    "mu": 0},
    }
    DEFAULT_N = 500
    DEFAULT_T = 1
    DEFAULT_VAL0 = 100
    
    
    def __init__(self, method=None, val0=None, T=None, N=None, params=None, **kwargs):
        if method is None: method = self.LOGNORM
        if not method in self.DEFAULT_PARAMS:
            raise ValueError(f"Unknown method {method}", tuple(self.DEFAULT_PARAMS))
        self.method=method
        if T is None: T = self.DEFAULT_T
        self.T=T
        if val0 is None: val0 = self.DEFAULT_VAL0
        self.val0=val0
        if N is None: N = self.DEFAULT_N
        self.N=N
        self.params=copy(self.DEFAULT_PARAMS[method])
        if not params is None: 
            for k,v in params.items():
                self.params[k] = v
        for k,v in kwargs.items():
                self.params[k] = v
        self._time = np.linspace(0, self.T, self.N+1)

    @property  
    def _new_gauss_vec(self):
        """
        creates a standard Gaussian vector of lenght self.N
        """
        return np.random.default_rng().normal(0, 1, self.N)
    
    def _lognormal(self):
        """
        creates a lognormal path
        """
        #print("[_lognormal] generating path")
        sig = self.params["sig"]
        dt = self.T / self.N
        sig_sqrt_dt = sig * sqrt(dt)
        mu_eff_dt = (-0.5*sig*sig + self.params["mu"])*dt
        logdvals = sig_sqrt_dt*self._new_gauss_vec + mu_eff_dt
        dvals = np.array([exp(x) for x in logdvals])
        dvals = np.concatenate(([self.val0], dvals))
        return np.cumprod(dvals)
    
    def _normal(self):
        """
        creates a normal path
        """
        #print("[_normal] generating path")
        dt = self.T / self.N
        sig_sqrt_dt = self.params["sig"] * sqrt(dt)
        mu_dt = self.params["mu"]*dt
        dvals = sig_sqrt_dt*self._new_gauss_vec + mu_dt
        #print("[_normal]", dt, sig_sqrt_dt, mu_dt, dvals)
        dvals = np.concatenate(([self.val0], dvals))
        return np.cumsum(dvals)
    
    def newpath(self):
        """
        creates a new random path vector
        """
        if self.method == self.LOGNORM:
            return self._lognormal()
        elif self.method == self.NORM:
            return self._normal()
        else:
            raise RuntimeError("How TF did we get here?!?")
    
    @property
    def time(self):
        """
        returns the time vector
        
        NOTE: the time vector is calculated initially and kept, and it is just returned
        here; the reason for this is that this vector can be stored with every generated
        path without risk of duplication
        """
        return self._time
    
    def generate(self, N, rtype=None):
        """
        generate paths
        
        :N:        number of paths to generate
        :retype:   return type (eg, list, tuple; None=generator)
        """
        result = (RPath(self.newpath(), self.time) for _ in range(N))
        if rtype is None:
            return result
        return rtype(result)

LOGNORM = RPathGen.LOGNORM 
NORM = RPathGen.NORM 
