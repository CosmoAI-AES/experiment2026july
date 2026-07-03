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
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Evaluation on the test set

This document evaluates one trained machine learning model,
see details below.
We use both conventional metrics and comparison with a 
resimulated image.

We use three datafiles, which must be downloaded if this
document is to be executed.
+ [testing.csv](../testing.csv) is ground truth for model training
+ [test.csv](test.csv) is the predicted amplitudes from machine learning.
+ [dataset.csv](../dataset.csv) is the original lens parameters used to generate
  the training, testing, and validation data, i.e. it has more
  rows than the other two sets.

The neural network used for machine learning is almost arbitrarily
chosen and has not been tuned.  The architecture is one of the
best performing in Nicolò's experiments on other datasets from
CosmoSim, but the hyperparameters are arbitrarily chosen.
The training set used is 16000 images.

The specification of the distribution is discussed in
[](Dataset.ipynb) and
can be downloaded ([dataset.toml](dataset.toml)).

+++

## Configuration

```{code-cell} ipython3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import IPython.display as idp
import toml
import json
import CosmoSim.Image as csimg
import CosmoSim as cs
import CosmoSim.datagen as dg
import CosmoSim.roulettegen as rg
imshow = csimg.imshow
```

Let us check the parameters underlying this test.
This gives the neural network and hyperparameters used.

```{code-cell} ipython3
cfg = toml.load( "ml.toml" )
print( "CosmoSim version is", cs.__version__ )
print( "Data from directory", cfg["output"]["directory"] )
print( "Neural network used:", cfg["settings"]["model"] )
print( json.dumps( cfg["hyperparameters"], indent=4 ) )
```

Let's load the test set, both the ground truth (`gt`) and the predictions (`df`).

```{code-cell} ipython3
gt = pd.read_csv( "../testing.csv", index_col="filename" )
df = pd.read_csv( "test.csv", index_col="filename" )
display( gt.head() )
display( df.head() )
```

As we can see, we have columns for all the parameters in question, and rows for the individual objects.  There is no superfluous data in the data frames.

We can also check the difference.

```{code-cell} ipython3
rawerrors = gt - df
display( rawerrors.head() )
```

That was straight forward.  Now we come to the tricky part - making sense of the numbers.
There are two questions that we want to consider.
1. How well does the model perform overall?
2. How do particular images behave, and what are the best and the worst samples?

+++

## Model evaluation

It is interesting to see how the model behaves on each column.
To see this, we can aggregate the rows, computing the mean of absolute errors and mean of squared errors.

```{code-cell} ipython3
mse = (rawerrors**2).mean()
mae = rawerrors.abs().mean()
mse.name = "MSE"
mae.name = "MAE"
colerrors = pd.concat( [ mse, mae ], axis=1 )
display( colerrors )
```

In practice, relative errors may be more interesting.
We can calculate that too.

```{code-cell} ipython3
colerrors["mean"] = gt.abs().mean()
colerrors["Relative MAE"] = colerrors["MAE"] / colerrors["mean"]
display( colerrors )
```

This looks good with mean relative errors less than $10^{-7}$, and no column is particularly bad or good.

+++

## Object evaluation

To investigate individual objects, we start by calculating sum of squared errors for each one.

```{code-cell} ipython3
sse = (rawerrors**2).sum(axis=1)
display(sse)
```

We can look at the three best and three worst images, thus,

```{code-cell} ipython3
sse.nlargest(3)
```

```{code-cell} ipython3
sse.nsmallest(3)
```

We note that the errors are small, but there is also a huge span between the best and the worst.
For the purpose of this test, we do not assume that we have access to the original images, but we can resimulate them from the roulette amplitudes.
First we record the filenames.

```{code-cell} ipython3
best = list(sse.nlargest(3).index)
worst = list(sse.nsmallest(3).index)
print( "Best:", best )
print( "Worst:", worst )
```

## Simulation from Roulette Amplitudes

Before we can simulate, we need to set up some basic parameters.

```{code-cell} ipython3
cfg = { "simulator" : { "imagesize" : 512, "cropsize" : 256, "xireference" : True }
      , "source" : { "mode" : "SersicSphere" } }
param = cs.Parameters( cfg )
```

::: {note}
The most critical parameter is the source mode.  Since the same model is
assumed for all objects, this is not included in the dataset for machine
learning and must be defined here.

The other parameters may be redundant, but the defaults are not
yet completely established, so they are there to be on the safe side.
+ `cropsize` is the final image size, and 256 is the size we normally
  use in our project.
+ `imagesize` needs to be at least equal to `cropsize`, but is otherwise
  generally only important when the image is translated (centred)
  before cropping.
+ `xireference : True`  uses roulette amplitudes from the apparent
  source position.  This is the default, and the alternative is not
  currently tested.
:::

Lets first consider the worst images and the ground truth.
The data is extracted as

```{code-cell} ipython3
gtw = gt.loc[ worst ]
display( gtw )
```

To simulate, we instantiate the `Resim` class which is the 
simulator for roulette resimulation.
For now, we test in just one image.

```{code-cell} ipython3
imsim = rg.Resim(gtw.iloc[0],param=param,verbose=0)
im = imsim.getImage()
imshow( im )
```

We can make the same simulation from the reconstructed amplitudes and compare.

```{code-cell} ipython3
dfw = df.loc[ worst ]
imsim2 = rg.Resim(dfw.iloc[0],param=param,verbose=0)
im2 = imsim2.getImage()
csimg.imageCompare( im, im2, "Ground Truth", "Reconstructed" )
```

This looks like perfect match.

We can do the same for all the top and bottom three.

```{code-cell} ipython3
for fn in worst:
    dfsim = rg.Resim(df.loc[fn],param=param,verbose=0)
    dfim = dfsim.getImage()
    gtsim = rg.Resim(gt.loc[fn],param=param,verbose=0)
    gtim = gtsim.getImage()
    csimg.imageCompare( dfim, gtim, fn, "Ground Truth", axiscross=True )
```

No visible discrepancy.  We can continue with the best images, obviously expecting perfect match again.

```{code-cell} ipython3
for fn in best:
    dfsim = rg.Resim(df.loc[fn],param=param,verbose=0)
    dfim = dfsim.getImage()
    gtsim = rg.Resim(gt.loc[fn],param=param,verbose=0)
    gtim = gtsim.getImage()
    csimg.imageCompare( dfim, gtim, "Reconstructed", "Ground Truth" )
```

## Simulations from Lens Parameters

We can also load the original lens parameters, from which 
the ground truth was computed.

```{code-cell} ipython3
orig = pd.read_csv( "../dataset.csv", index_col="filename" )
display( orig.head() )
```

This dataset is different from the others, giving physical parameters
of the lens instead of roulette amplitudes in a point of observation.
Thus, we need a different simulator.  On a positive note, we can use
raytrace simulation, which is accurate.

The `SimImage` simulator is parameterised in a slightly different
way.  We need to add the row data from the dataset to the 
`Parameters` object instead of passing it as a separate argument.
To avoid interference, we make a copy of `params`.

```{code-cell} ipython3
cfg["simulator"]["model"] = "Raytrace"
cfg["simulator"]["centred"] = False
cfg["lens"] = { "mode" : "SIE" }
p2 = cs.Parameters( cfg )
for fn in worst:
    dfsim = rg.Resim(df.loc[fn],param=param,verbose=0)
    dfim = dfsim.getImage()
    p2.setRow( orig.loc[fn] )
    gtsim = dg.SimImage(param=p2,verbose=0)
    gtim = gtsim.getImage()
    csimg.imageCompare( dfim, gtim, fn, "Original raytrace simulation", axiscross=True )
```

The match is not perfect, but we do see that the primary image is correctly placed. The shape is not accurate on the edges. 
It is significant that the shadow in the difference image around the primary image is either black or white, never both.
This means that the image is smaller or larger, and not rotated or awfully misshaped in general.
Clearly, this is a limitation of the roulette formalism, since there is no difference between reconstructed and ground truth simulation.

```{code-cell} ipython3
for fn in best:
    dfsim = rg.Resim(df.loc[fn],param=param,verbose=0)
    dfim = dfsim.getImage()
    p2.setRow( orig.loc[fn] )
    gtsim = dg.SimImage(param=p2,verbose=0)
    gtim = gtsim.getImage()
    csimg.imageCompare( dfim, gtim, fn, "Original raytrace simulation", axiscross=True )
```

Interestingly, the best images do not perform any better than the worst in terms of visual comparison between roulettes and raytrace.

+++

## Conclusion

+++

We see that this machine learning model make accurate prediction as far as optical perception goes, but there are limitations to the roulette representations.

This means that there is nothing to gain from further research on machine learning models at this stage.
