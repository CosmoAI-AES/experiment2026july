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

# Sample Datasets

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
| Source size | $\sigma$ | `sigma` | Uniform | $0.04"\le f\le 0.2"$ |
| Sersic index | $n_s$ | `n_sersic` | Uniform |  $1\le n_s\le 5$ |
| Luminosity  | $l$ | `luminosity` | Exponential |  $10\le l\le 20$, $\lambda=2.0$ |

1.  The source position is given in polar co-ordinates $(R,\phi)$.
    + The distance $R$ is chosen to be inside or around the critical curve,
      hence the limit $R\le c\theta_E$ for some constant $c$.
2.  Angles are given in degrees in the source code and chosen uniformly from a
    half circle, because of symmetry.
3.  The source is spherical with a sersic profile.

::: {note} Definition
The **exponential distribution** returns
$$u = 1 - e^{-\lambda\cdot x}$$
where $x$ is exponentially distributed, that is with a probability 
density function 
$$f(x;\lambda) = \lambda e^{-\lambda x}$$
for positive $x$.
This $u$ is scaled to within the given range.
:::

::: {note} Cluster lenses
For cluster lenses, each constituent lens is placed in the same way as 
the source, relative to the origin, in polar co-ordinates $(R_L,\phi_L)$.
+ select random distance $R_L\le2\theta_E$, where $\theta_E$ is the
  Einstein radius of the constituent lens
+ select random angle $\phi_L$ 
:::

+++

## Dataset generation

Before we start, lets check the version.

```{code-cell} ipython3
import CosmoSim
print( CosmoSim.__version__ )
```

The `CosmoSim.dataset` module provides the functions to generate random datasets.
We will also use the Imge module as before.

```{code-cell} ipython3
import CosmoSim.dataset as csd
import CosmoSim.Image as csimg
```

The probability distribution is specified by a nested `dict`, typically 
defined in a TOML file, similar to those used in other CosmoSim modules.
However, the `dataset` submodule does not use the `Parameters` class for
its parameters at present.

```{code-cell} ipython3
cfg = csd.readtoml( "dataset.toml" )
display( cfg )
```

We see that this configuration is set up to generate 24000 images in 256$\times$256 format.
We can see the paremeter ranges for the lens and for the source, which is a sphere with sersic profile.
As of CosmoSim v3.1 we can still use the config parameter for both the simulation model and the lens;
"raysie" is raytrace simualation of a SIE lens.
This will change in v3.2, to require separate settings for `simulator.model` and `lens.mode`.

::: {tip}
Download [dataset.toml](./dataset.toml).
:::

::: {note} Remark
For the most part, this TOML file allows the same settings as the TOML file
for other CosmoSim modules.  When a parameter is to be drawn uniformly at 
random, the range is given by two parameters, suffixed by `-min` and `-max`
respectively.  Thus, when there is an `einstein` parameter, we can define
`einstein-min` and `einstein-max`.  In the case of luminosity, there is also
a `luminosity-lambda` parameter for the exponential distribution.

For string values, such as for `simulator.config`, a list of 
strings can be given, to draw one element uniformly at random.
:::

To get a single random object, we use the `getline()` function.

```{code-cell} ipython3
ob = csd.getline( cfg, fn="test.png" )
display( ob )
```

The filename `fn` is used for indexing as well as saving images in 
batch processing.  
It can otherwise be omitted.

+++

## Generating the image

Observe that the row generated by `getline()` is a flat structure,
and not the hierarchical structure used in TOML files.
However, we can add the parameters from `ob` to the parameters
object as follows.

```{code-cell} ipython3
import CosmoSim as cs
import CosmoSim.datagen as csg
import matplotlib.pyplot as plt

param = cs.Parameters( )
param.setRow( ob )
```

Now we can run the simulator as we have done in previous demos
(e.g. [](../Demo01.ipynb).

```{code-cell} ipython3
imsim = csg.SimImage( param, verbose=0 )
im = imsim.getImage()
csimg.imshow( im, "Random image test" )
```

The image is very small, but that may well be correct.
We can inspect the undistorted source image as well.

```{code-cell} ipython3
actual = imsim.getActualImage()
csimg.imshow( actual, "Undistorted source" )
```

## Sampling

To get an impression of distribution, we can bulk generate
configurations.  Let's first look at the dataset, as follows.

```{code-cell} ipython3
import pandas as pd
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

::: {warning}
There is a memory leak in the C++ code which threatens to crash
the computer if the simulator is instantiated several hundred times.
This can be avoided by instantiating the backend simulator only once.
and pass it as an argument
```
sim = CosmoSim.CosmoSim()
imsim = csg.SimImage( param, sim=sim, verbose=0 )
```
This is of course a bug that we will try to resolve.  
The scripts (below) avoid the issue by instantiating the backend 
simulator only once.
Interactive applications are unlikely to instantiate enough instances
to make a problem.
:::

## Bulk generation

Managing thousands of images in a Jupyter Notebook is not very practical.
In most cases, one will want to generate the dataset from the command line
```
python -m CosmoSim.dataset config.toml dataset.csv
```
This generates a CSV file which can be used for image generation.

It is also possible to generate the dataset and the images in one go,
using
```
python -m CosmoSim --rnd --toml config.toml --csvfile dataset.csv \
      --outfile roulette.csv --directory images
```
This generates the random dataset (dataset.csv) and the images in
the images directory, as well as a dataset of corresponding
roulettes amplitudes (roulette.csv).
