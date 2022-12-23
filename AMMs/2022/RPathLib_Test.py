# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from matplotlib import pyplot as plt
from rpathlib import *

# # RPathLib Test and Demo

# ## RPathLib [TEST]

PG = RPathGen(method=RPathGen.LOGNORM, sig=0.5)
assert len(PG._gauss_vec()) == PG.N
assert PG.time is PG.time

PG = RPathGen(method=RPathGen.LOGNORM, sig=0.2, N=1000)
for _ in range(50):
    plt.plot(PG.newpath(), color="grey")

p = RPath(PG.newpath(), PG.time)

for skp,ofs, col in [(0,0,"lightblue"), (200,0,"green"), (200,100,"red")]:
    plt.plot(p.time(skip=skp, offset=ofs), p.path(skip=skp, offset=ofs), color=col, label=f"s={skp}, o={ofs}")
plt.legend()
plt.grid()

for skp,ofs, col in [(0,0,"lightblue"), (20,0,"green"), (125,0,"red")]:
    plt.plot(p.time(skip=skp, offset=ofs), p.path(skip=skp, offset=ofs), color=col, label=f"s={skp}, o={ofs}")
plt.legend()
plt.grid()
    

pl = list(PG.generate(50))

for p in pl:
    plt.plot(p.time0, p.path0, color="grey")


