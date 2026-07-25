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

# Sample Datasets for SIE

The purpose of this demonstration is to explore reasonable parameters
for dataset generation.
We aim to follow established custom in the literature, as far as possible. 

::: {note} Image scale
Hezaveh uses $192\times192$ image size with a pixel corresponding
to 0.04" (seconds of arc).
We will use the same scale of 0.04"/pixel, but use slightly larger
images.
Calculations are made in 
$512\times512$ and crop to $256\times256$
afterwards.
:::

Because we centre the images on the visible light, there will usually
be a lot of empty background to crop.  Making the calculations on  
larger images will prevent many cropping artifacts.
We also note that the image size is limited by computer memory, when
they are used with machine learning.  We were unable to use
$512\times512$ images on a GPU with 50Gb memory, but $256\times256$ work.

## Parameter ranges

To build training sets we need to generate random sets of plausible 
images.  The ranges and probability distribution may vary from study
to study.  The parameters we establish here are designed to be realistic
examples of strong lensing, erring on the side of wider ranges.

| Parameter | Symnol | Identifier | Distribution | Range |
| :- | :- | :- | :- | :- |
| Einstein radius | $\theta_E$ | `einsteinradius` | Uniform | $0.1"\le\theta_E\le3.0"$ |
| Source position | $R$ | `position.r`  | Uniform | $R\le1.2\cdot\theta_E$ |
| Source location | $\phi$ | `position.phi` | Uniform | $0\le\phi\le180$ |
| Lens orientation | | `orientation`  | Uniform | $0\ldots180$ |
| Source orientation | |  | Uniform | $0\ldots180$ |
| Lens ellipticity | $f$ | `ellipseratio` | Uniform | $0.1\le f\le 0.9$ |
| Source size | $\sigma$ | `sigma` | Uniform | $0.2"\le f\le 2.0"$ |
| Sersic index | $n_s$ | `n_sersic` | Uniform |  $1\le n_s\le 5$ |
| Luminosity  | $l$ | `luminosity` | Exponential |  $20\le l\le 80$, $\lambda=2.0$ |

1.  The source position is given in polar co-ordinates $(R,\phi)$.
    + The distance $R$ is chosen to be inside or around the critical curve,
      hence the limit $R\le c\theta_E$ for some constant $c$.
2.  Angles are given in degrees in the source code and chosen uniformly from a
    half circle, because of symmetry.
3.  The source is spherical with a sersic profile.

::: {note} Further reading
The dataset generation is outlined in more detail in
[](xref:cosmoai/dataset-1).
Her we focus on the distribution used in this particular experiment.
:::

+++

## Dataset generation

Before we start, we import the required modules and check the CosmoSim
version.

```{code-cell} ipython3
import CosmoSim
import CosmoSim.dataset as csd
import CosmoSim.Image as csimg
import CosmoSim as cs
import CosmoSim.datagen as csg
import matplotlib.pyplot as plt
import pandas as pd

print( CosmoSim.__version__ )
```

The `CosmoSim.dataset` module provides the functions to generate random datasets.
The probability distribution is specified by a nested `dict`, typically 
defined in a TOML file, similar to those used in other CosmoSim modules.
However, the `dataset` submodule does not use the `Parameters` class for
its parameters at present.

::: {tip}
Download [sie-dataset.toml](./sie-dataset.toml).
:::

```{code-cell} ipython3
cfg = csd.readtoml( "sie-dataset.toml" )
display( cfg )
```

We see that this configuration is set up to generate 15000 images in
256$\times$256 format.
We can see the parameter ranges for the lens and for the source,
which is a sphere with sersic profile.
It is set up to do raytrace simulation with a SIE lens.

+++

## Sampling the distribution

To get an impression of distribution, we can bulk generate
configurations.  Let's first look at the dataset, as follows.

```{code-cell} ipython3
obs = [ csd.getline( cfg ) for ob in range(8) ]
df = pd.DataFrame( obs )
display(df)
```

To create the images we make a quick function to generate a each one.

```{code-cell} ipython3
def getImage(ob):
    param = cs.Parameters(cliconfig=cfg)
    param.setRow( ob )
    sim = csg.SimImage( param, verbose=0 )
    im = sim.getImage() 
    csimg.crop( im, param.get( "cropsize" ), verbose=0 )
    return im
```

Each line of the function is as used above in the document,
except that we have added cropping.
Now we can quickly generate a list of images and display them.

```{code-cell} ipython3
import CosmoSim.Image as csimg

ims = [ getImage(ob) for ob in obs ]

fig = plt.figure(figsize=(20, 10))
fig.tight_layout(pad=0.0)
plt.subplots_adjust(hspace=0.1, wspace=0.1) 

for idx,im in enumerate(ims):
    fig.add_subplot(2, 4, idx+1)
    csimg.imshow( im )
```

## Closure

Using this dataset, we have trained a machine learning model,
using $10\,000$ images for training.  

Continue to [](experiment001/Testing.ipynb).
