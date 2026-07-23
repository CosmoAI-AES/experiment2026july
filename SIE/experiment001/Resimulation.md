---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Roulette Simulation (Experiment July 2026)

We build on [](xref:cosmoai/demo03resimulation/).
The following modules are needed.

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

This test uses the features available in CosmoSim v3.2.4.
We make a separate to make tests enabled by future, unreleased
versions: [](Resimulation02.ipynb).

We will also need the following files, which will be loaded by 
the code:
+ [sie-dataset.toml](../sie-dataset.toml).
+ [sie-dataset.csv](../sie-dataset.csv).
+ [sie-testing.csv](../sie-testing.csv).
+ [pred-sie-testing.csv](./experiment001/pred-sie-testing.csv).

## Review of the SIE experiment data

We load and compare the testing results and the ground truth.

```{code-cell} ipython3
gt = pd.read_csv( "../sie-testing.csv", index_col="filename" )
pr = pd.read_csv( "pred-sie-testing.csv", index_col="filename" )
rawerrors = gt - pr
sse = (rawerrors**2).sum(axis=1)
```

From the dataset, we pick the three best and the three worst data points.

```{code-cell} ipython3
best = list(sse.nlargest(3).index)
worst = list(sse.nsmallest(3).index)
print( "Best:", best )
print( "Worst:", worst )
```

## Simulation from lens parameters

To simulate the images, we need to load the parameters from
`sie-dataset.csv`.
From this dataset we pick just the selected rows.

```{code-cell} ipython3
df = pd.read_csv( "../sie-dataset.csv", index_col="filename" )
df = df.loc[best+worst]
display( df )
```

We will also need the simulator configuration from `sie-dataset.toml`.

```{code-cell} ipython3
with open( "../sie-dataset.toml", 'rb' ) as f:
            toml = tl.load(f)
param = Parameters( toml )
```

Now we can run the simulator.

```{code-cell} ipython3
param.setRow( df.iloc[0] )
param["simulator"]["centred"] = False
imsim0 = SimImage( param, verbose=0 )
ray0 = imsim0.getImage()
param["simulator"]["model"] = "Roulette" 
rousim0 = SimImage( param, verbose=0 )
rou0 = rousim0.getImage()
csimg.imageCompare( ray0, rou0, "Raytrace", "Roulette" ) 
```

We can also get annotations on the images, like this:

```{code-cell} ipython3
param.setRow( df.iloc[0] )
param["simulator"]["centred"] = False
ray0a = rousim0.getAnnotated(convergenceRing=None,centrePoint=None)
param["simulator"]["model"] = "Roulette" 
rou0a = rousim0.getAnnotated(convergenceRing=None,centrePoint=None)
csimg.imageCompare( ray0a, rou0a, "Raytrace", "Roulette" ) 
```

The roulette simulation seems perfect within the convergence ring.

::: {warning}
The call to `setRow()` overrides settings which have been set
manually.  Hense we need to reset `simulator.centred` each time,
but we do not have to reset `simulator.model` to Raytrace after the
roulette simulation.
:::

+++

## The full image set

We can repeat the simulation on all the selected imags.

```{code-cell} ipython3
for index, row in df.iterrows():
    param.setRow( row )
    param["simulator"]["centred"] = False
    ray = SimImage( param, verbose=0
        ).getAnnotated(centrePoint=None)
    param["simulator"]["model"] = "Roulette" 
    rou = SimImage( param, verbose=0
        ).getAnnotated(centrePoint=None)
    csimg.imageCompare( ray, rou, index, "Roulette", axiscross=True ) 
```

Here we observe very good match between the roulette and raytrace
simulations, except possibly where the visible image is tiny and centred,
where it is difficult to judge.

+++

## Resimulation

+++

It will be interesting to check if resimulation confirms the result.
First a simple check with a single data point.

```{code-cell} ipython3
row = imsim0.getData()
display( row )
```

```{code-cell} ipython3
rp = Parameters( { "simulator" : { "cropsize" : 256 } } )
resimImage = Resim(  row, rp, verbose=0 ).getImage()
csimg.imageCompare( resimImage, rou0, "Resimulation", "Original Roulette" )
csimg.imageCompare( resimImage, ray0, "Resimulation", "Original Raytrace" )
```

Possibly confusing, the larger of the distorted images fall outside the convergence ring. The primary image on which the roulette representation focuses, is comparatively small.

We see a small discrepancy in this primary image. This may or may not be significant in itself, but it is disconserting because the original roulette simulation was perfect.

Most of the other images show similar discrepancies, but some are also alright.

```{code-cell} ipython3
for index, row in df.iterrows():
    param.setRow( row )
    param["simulator"]["centred"] = False
    imsim = SimImage( param, verbose=0 )
    ray = imsim.getImage()
    rr = imsim.getData()
    rou = Resim( rr, rp, verbose=0 ).getImage()
    csimg.imageCompare( ray, rou, "Raytrace", "Roulette", axiscross=True ) 
```

Note that we have simulated without centring.

```{code-cell} ipython3
print( "Centring:", param.get( "centred" ) )
```

## Centred mode

Let us reset the simulator to use centred mode.

```{code-cell} ipython3
with open( "../sie-dataset.toml", 'rb' ) as f:
            toml = tl.load(f)
param = Parameters( toml )
```

```{code-cell} ipython3
for index, row in df.iterrows():
    param.setRow( row )
    imsim = SimImage( param, verbose=0 )
    ray = imsim.getImage()
    rr = imsim.getData()
    rou = Resim( rr, rp, verbose=0 ).getImage()
    csimg.imageCompare( ray, rou, "Raytrace", "Roulette", axiscross=True ) 
```

We see a similar discrepancy in centred mode.

+++

## Validation of the generated amplitudes

+++

Since we cannot explain the discrepancy, we have to test every conceivable variation until we can find a cause. Now let's check the amplitudes generated here against the file generated by the batch run.  Again we pick only the selected rows.

```{code-cell} ipython3
gt = pd.read_csv( "../sie-roulette.csv", index_col="filename" )
gt = gt.loc[best+worst]
display( gt )
```

We did not store the amplitudes from `getData` when we did the simulation, so we need to create a DataFrame of these amplitudes now.

```{code-cell} ipython3
l = []
for index, row in df.iterrows():
    param.setRow( row )
    imsim = SimImage( param, verbose=0 )
    l.append( imsim.getData() )
gen = pd.DataFrame( l )
display( gen )
```

The interesting step is now to compare `gen` to the ground truth `gt`.

```{code-cell} ipython3
diff = gen.drop("source",axis=1)-gt
display( diff )
```

This looks like a good match, but to make sure we do not miss anything, we can look at the maximum absolute differences.

```{code-cell} ipython3
display( diff.abs().max() )
```

Right, so differences are around $10^{-14}$ and smaller.

+++

## Closure

This comparison shows good match between raytrace and roulette, except possibly for small images close to the origin.

What is confusing in these images is that the spurious images appear to be off compared to the convergence ring.  It is possible that the definition we use for the convergence ring may only be valid for symmetric (spherical) lenses.

```{code-cell} ipython3

```
