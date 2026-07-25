---
authors:
- name: Hans Georg Schaathun
exports:
- format: pdf
  output: SIE-Resimulation.pdf
  template: lapreprint
jupytext:
  notebook_metadata_filter: all,-language_info,-jupytext.text_representation.jupytext_version
  cell_metadata_filter: tags
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
parts:
  abstract: This is work in progress.  We are debugging the report format.
title: Resimulation from the SIE Experiment
---

# Roulette Resimulation (Experiment July 2026)

This document will investigate inaccuracies in the standard evaluation
document, [](Testing.ipynb).
We hypothesised that image centring causes minor numerical inaccuracy, 
which differs between the original raytrace image and the roulette resimulation.

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

::: {warning} Version requirement
This report assumes the features introduced in CosmoSim v3.3.
Using v3.2.4, the code will run, but the discrepnacies reported
in the last test will also be present in the previous tests
using the `Resim` class.
:::

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
display( df.T )
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
raysim0 = SimImage( param.copy(), verbose=0 )
ray0 = raysim0.getImage()
param["simulator"]["model"] = "Roulette" 
rousim0 = SimImage( param, verbose=0 )
rou0 = rousim0.getImage()
csimg.imageCompare( ray0, rou0, "Raytrace", "Roulette" ) 
```

Here we cannot see the discrepancy that we found in [](Testing.ipynb).

+++

::: {warning}
We instantiate `raysim0` with `param.copy()` to make sure that we can reuse `raysim0` later with the same result.  Since `SimImage` does not copy the `Parameters` object, its configuration will change if the object is subsequently changed.
:::

We can also get annotations on the images, like this:

```{code-cell} ipython3
param.setRow( df.iloc[0] )
param["simulator"]["centred"] = False
ray0a = raysim0.getAnnotated(convergenceRing=None,centrePoint=None)
param["simulator"]["model"] = "Roulette" 
rou0a = rousim0.getAnnotated(convergenceRing=None,centrePoint=None)
csimg.imageCompare( ray0a, rou0a, "Raytrace", "Roulette" ) 
```

The roulette simulation seems perfect within the convergence ring.

::: {warning}
The call to `setRow()` overrides settings which have been set
manually.  Hence we need to reset `simulator.centred` each time,
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
    plt.savefig( f"annot1-{index}" )
```

These images show good match between raytrace and roulette, although it may be hard to judge when the visible image is small and close to the origin. 
Where the primary image is far from the origin, the match looks perfect.
It may be somewhat easier to see without the annotations.

```{code-cell} ipython3
for index, row in df.iterrows():
    param.setRow( row )
    param["simulator"]["centred"] = False
    ray = SimImage( param, verbose=0
        ).getImage()
    param["simulator"]["model"] = "Roulette" 
    rou = SimImage( param, verbose=0
        ).getImage()
    csimg.imageCompare( ray, rou, index, "Roulette", axiscross=True ) 
    plt.savefig( f"resim1-{index}" )
```

## Resimulation

+++

It will be interesting to check if resimulation confirms the result.
First a simple check with a single data point.

```{code-cell} ipython3
row = raysim0.getData()
display( row )
```

```{code-cell} ipython3
newcfg = { "simulator" : { "cropsize" : 256, "nterms" : 4, "centred" : False }
         , "source" : { "mode" : "SersicSphere" }
         } 
rp = Parameters( newcfg )
resimImage = Resim(  row, rp, verbose=0 ).getImage()
csimg.imageCompare( resimImage, rou0, "Resimulation", "Original Roulette" )
csimg.imageCompare( resimImage, ray0, "Resimulation", "Original Raytrace" )
```

This looks perfect as far as the roulette formalism goes, as do the other selected images.

::: {note} Primary image
Possibly confusing, the larger of the distorted images fall outside the convergence ring. The primary image on which the roulette representation focuses, is comparatively small.
:::

```{code-cell} ipython3
for index, row in df.iterrows():
    param.setRow( row )
    param["simulator"]["centred"] = False
    imsim = SimImage( param, verbose=0 )
    ray = imsim.getImage()
    rr = imsim.getData()
    rou = Resim( rr, rp, verbose=0 ).getImage()
    csimg.imageCompare( ray, rou, "Raytrace", "Roulette", axiscross=True ) 
    plt.savefig( f"resim2-{index}" )
```

Note that we have simulated without centring.

```{code-cell} ipython3
print( "Centring:", param.get( "centred" ) )
```

## Centred mode

Let us reset the simulator to use centred mode.
Here we will see the numeric inaccuracy.

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
    csimg.imageCompare( ray, rou, index, "Roulette", axiscross=True ) 
    plt.savefig( f"centre-{index}" )
```

This is also perfect.

+++

## Validation of the generated amplitudes

+++

During the debugging phase, we also check the amplitudes generated here against the file generated by the batch run.  Although this did not give any answers at the time, we give the code in case it may be useful in the future.

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
It also demonstrates that the new mode of operation in v3.3 is a necessary improvement.

Using roulette resimulation to validate machine learning predictions, we have to expect a numeric inaccuracy, due to the centring of the images subjected to the machine learning model.
This is probably not a problem in practice, since observational images will show all sorts of inaccuracies in any event.

```{code-cell} ipython3

```
