---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.4
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Roulette Simulation (Experiment July 2026)

We build on [](xref:cosmoai/demo/Demo03Resimulation.ipynb).

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tomllib as tl

from CosmoSim.datagen import SimImage
import CosmoSim.Image as csimg
from CosmoSim import Parameters
from CosmoSim.roulettegen import Resim

import CosmoSim as cs
print( "CosmoSim version", cs.__version__ )
```

## Review of the SIE experiment data

Files used:
+ [SIE/sie-dataset.csv](./SIE/sie-dataset.csv).
+ [SIE/sie-testing.csv](./SIE/sie-testing.csv).
+ [SIE/pred-sie-testing.csv](./SIE/experiment001/pred-sie-testing.csv).

We load and compare the testing results and the ground truth.

```{code-cell} ipython3
gt = pd.read_csv( "SIE/sie-testing.csv", index_col="filename" )
pr = pd.read_csv( "SIE/experiment001/pred-sie-testing.csv", index_col="filename" )
rawerrors = gt - pr
sse = (rawerrors**2).sum(axis=1)
```

From the dataset, we pick the three best and the three worst data points.

```{code-cell} ipython3
best = list(sse.nlargest(3).index)
worst = list(sse.nsmallest(3).index)
print( best )
print( worst )
```

We load the lens parameters and pick the rows corresponding to these
best and worst images.

```{code-cell} ipython3
df = pd.read_csv( "SIE/sie-dataset.csv", index_col="filename" )
df = df.loc[best+worst]
display( df )
```

```{code-cell} ipython3
with open( "SIE/sie-dataset.toml", 'rb' ) as f:
            toml = tl.load(f)
param = Parameters( toml )
```

```{code-cell} ipython3
param.setRow( df.iloc[0] )
param["simulator"]["centred"] = False
imsim = SimImage( param, verbose=0 )
ray0 = imsim.getImage()
param["simulator"]["model"] = "Roulette" 
rou0 = SimImage( param, verbose=0 ).getImage()
csimg.imageCompare( ray0, rou0, "Raytrace", "Roulette" ) 
```

```{code-cell} ipython3
param.setRow( df.iloc[0] )
param["simulator"]["centred"] = False
ray = SimImage( param, verbose=0 ).getAnnotated(convergenceRing=None,centrePoint=None)
param["simulator"]["model"] = "Roulette" 
rou = SimImage( param, verbose=0 ).getAnnotated(convergenceRing=None,centrePoint=None)
csimg.imageCompare( ray, rou, "Raytrace", "Roulette" ) 
```

## The full image set


```{code-cell} ipython3
for index, row in df.iterrows():
    param.setRow( row )
    param["simulator"]["centred"] = False
    ray = SimImage( param, verbose=0
        ).getAnnotated(centrePoint=None)
    param["simulator"]["model"] = "Roulette" 
    rou = SimImage( param, verbose=0
        ).getAnnotated(centrePoint=None)
    csimg.imageCompare( ray, rou, "Raytrace", "Roulette", axiscross=True ) 
```

## Resimulation

```{code-cell} ipython3
df2 = gt.iloc[best+worst]
row = imsim.getData()
resimImage = Resim(  row, param, verbose=0 ).getImage()
csimg.imageCompare( resimImage, rou0, "Resimulation", "Original Roulettte" )
```

```{code-cell} ipython3
for index, row in df.iterrows():
    param.setRow( row )
    param["simulator"]["centred"] = False
    imsim = SimImage( param, verbose=0 )
    ray = imsim.getAnnotated(centrePoint=None)
    param["simulator"]["model"] = "Roulette" 
    rr = imsim.getData()
    rou = Resim( rr, param, verbose=0
        ).getAnnotated(centrePoint=None)
    csimg.imageCompare( ray, rou, "Raytrace", "Roulette", axiscross=True ) 
```

## Closure

This comparison shows good match between raytrace and roulette, except possibly for small images close to the origin.

What is confusing in these images is that the spurious images appear to be off compared to the convergence ring.  It is possible that the definition we use for the convergence ring may only be valid for symmetric (spherical) lenses.

```{code-cell} ipython3

```
