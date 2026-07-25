---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Cluster Dataset (2x SIE)

In this demo we show the generation of a dataset for machine
learning sporting two SIE lenses.
To run the notebook, you will have to download several datafiles
from [](Cluster2.md).

::: {warning} 
This demo uses a feature from CosmoSim v3.2, taking the Critical Curve
into account when placing the sources.  This will not work in v3.1.
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
from CosmoSim import Parameters
```

The distribution of the dataset is configured as follows.

```{code-cell} ipython3
cfg = csd.readtoml( "cluster2-dataset.toml" )
display( json.dumps( cfg ) )
```

Each constituent lens is placed in a random direction from the origin,
at a random distance upper bounded as $c\theta_E$ where $\theta_E$ is
the Einstein radius and $c$ is the constant given as `cluster.maxrelativelocation`.

We can draw a random object as before.

## A sample

Let us make a random sample for review.
Firstly, we make a convenience function to run one simulation and
retrieve the image.

```{code-cell} ipython3
def mkimg(ob):
      p0 = Parameters( )
      p0.setRow( ob )
      sim = SimImage( p0, verbose=0 )
      return sim.getImage()
```

Using this function, we can make a list of simulations, and plot the
results.

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

## Annotations

It may be interesting to look at the Critical Curves, to get an idea of the shape of the lenses.
Instead of `mkimg()` we make a function to get annotated images.

```{code-cell} ipython3
def mkannotation(ob):
      p0 = Parameters( )
      p0.setRow( ob )
      sim = SimImage( p0, verbose=0 )
      return sim.getAnnotated()
```

We use the same datasets as before, but generate new and annotated images.

```{code-cell} ipython3
ims = [ mkannotation(ob) for ob in obs ]
csimg.showImages( ims, size=(2,4), titles=ts )
```

The yellow curve is the Critical Curve, and the blue circle is the convergence
ring of the roulette formalism.
The red point is the centre of light.

## Closure

We have attemmpted to design the dataset so that it displays interesting
samples of strong lensing.  
More research is needed on actual distributions of observed lenses, and
how a representative sample can be designed for machine learning.

```{code-cell} ipython3

```
