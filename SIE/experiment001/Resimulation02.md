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

# Testing different approaches to Roulette Resimulation 

This demo further explores the inaccuracies identified in
[](Resimulation.ipynb).
We use the same modules.

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

We require at least v3.3.0b1. 
With  v3.2.4 we can only reproduce previous results.

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

## Loading lens parameters

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

Simulations from lens parameters are found in [](Resimulation.ipynb).

+++

## Resimulation

+++

CosmoSim v3.3 offers a new configuration parameter,
`resimulation.drawmode`.
If we set this to `"xi"` the distorted image is drawn
directly in the corrected image.
In the default mode, the image is translated after simulation.
Thus we define a new `Parameters` object.

```{code-cell} ipython3
cfg = { "simulator" : { "cropsize" : 256 }
      , "resimulatoion" : { "drawmode" : "xi" }
      }
rp = Parameters( cfg )
```

Now we can run he simulation as we did before.


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

This removes the error observed in previous resimulation.

## Conclusion

This test shows that the new "xi" drawmode gives better consistency
between simulation modes.
CosmoSim v3.3 should be updated to use this mode as default.

