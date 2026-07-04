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

# Cluster Lenses (Demo n° 2)

In this demo we show the generation of a dataset for machine
learning sporting two SIE lenses.
We assume that the principles are known from [](/demo/Cluster01.ipynb).

::: {warning} 
This demo uses a feature from CosmoSim v3.2, taking the Critical Curve
into account when placing the sources.  This will not work in v3.1.
:::

::: {warning}
Work in progress
:::

## Preparation

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from CosmoSim.datagen import SimImage
import CosmoSim.Image as csimg
import CosmoSim.dataset as csd
from CosmoSim import CosmoSim, Parameters
```

## First test of Cluster Lenses

We choose the following parameters for the first simulation.

```{code-cell} ipython3
cfg = { "simulator" : { "model" : "Raytrace", "cropsize" : 256 }
      , 'lens': { "cluster" : "SIE/5/5/8/0.3/45;SIE/-5/-5/8/0.3/45" }
      , 'source': {
            'mode': 'Spherical',
            'sigma': 5,
            'theta': 45,
            'position': 'cartesian'}
      , 'position': { 'x': 11.01, 'y': 0.31 }
      }
param = Parameters( cfg )
```

The new element is the lens specification, which is not as user friendly as it was.
The `cluster` attribute gives a list of lenses separated by semicolon.
Each constituent lens consists of the lens model (`"SIE"`) and a list of
lens parameters separated by slash (`/`).  For SIE, these parameters are
$x$/$y$/$\theta_E$/$f$/$\phi_L$, i.e. position $(x,y)$, Einstein radius,
ellipticity, and orientation.

Given the parameters, the simulation is as before.

```{code-cell} ipython3
imsim = SimImage( param, verbose=0 )
im = imsim.getImage()
csimg.imshow( im, title="First example of a cluster lens" )
```

## Random dataset

We can also generate random datasets.

```{code-cell} ipython3
cfg = csd.readtoml( "cluster.toml" )
display( json.dumps( cfg ) )
```

::: {tip}
Download [cluster.toml](./cluster.toml).
:::

::: {note} Remark
The TOML file is similar to the one used in [](/demo/Dataset.ipynb).
The most notable difference is the `[cluster]`.
We also have to specify the lens model for the constuent lenses with the
`lens.model` parameter.
It cannot be inferred from `simulator.config`.
:::

Each constituent lens is placed in a random direction from the origin,
at a random distance upper bounded as $c\theta_E$ where $\theta_E$ is
the Einstein radius and $c$ is the constant given as `cluster.maxrelativelocation`.

We can draw a random object as before.

```{code-cell} ipython3
ob = csd.getline( cfg, fn="test.png" )
display( ob )
```

```{code-cell} ipython3
param = Parameters( )
param.setRow( ob )
imsim = SimImage( param, verbose=2 )
im = imsim.getImage()
csimg.imshow( im )
```

## A sample

```{code-cell} ipython3
def mkimg(ob):
      p0 = Parameters( )
      p0.setRow( ob )
      sim = SimImage( p0, verbose=0 )
      return sim.getImage()
```

```{code-cell} ipython3
obs = [ csd.getline( cfg ) for _ in range(8) ]
ims = [ mkimg(ob) for ob in obs ]
ts = [ f"Image {i}" for i in range(8) ]
csimg.showImages( ims, size=(2,4), titles=ts )
```

If we take a particular  interest in one particular image, say no 1, we can easily inspect its parameters.

```{code-cell} ipython3
print( obs[1] )
```

The cluser specification does not show in the row view, but we can single that one out to see properly.

```{code-cell} ipython3
print( obs[1]["cluster"] )
```

## Closure
