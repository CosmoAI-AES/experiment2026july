---
jupytext:
  formats: ipynb,md:myst
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

# Training and testing a single model.

It is possible to run machine learning through Jupyter Notebook,
but I would not recommend it.  It takes a long time, and it is
generally better to run it in the background, from the command line.
here, I will outline the process as I have used it.

**Requirement**
This demo requires CosmoSim v3.1 and droulette v0.1.

**Step 1.** Generate a dataset
This is discussed in detail in [](Dataset.ipynb).

A typical command looks like this:
```sh
time python -m CosmoSim --toml dataset.toml --rnd \
         --csvfile dataset.csv --outfile roulette.csv  \
         --directory images 
```
This is the sample data dataset configuration: [dataset.toml](./dataset.toml).

**Step 2.** Prepare the dataset for machine learning.

For machine learning, we need separate datasets for training, validation, and
testing, which are split from the base dataset.  We may also need roulette
simulations to use a ground truth for validation.  We generate these datasets
with the `droulette` package.
```sh
python -m droulette.split problem.toml
```
The sample file is [problem.toml](./problem.toml). We can have a look at it:

```{code-cell} ipython3
import json, tomllib as tl
with open( "problem.toml", 'rb') as f:
            toml = tl.load(f)
print( json.dumps( toml, indent=4 ) )
```

Observe that file and directory names are specified in the config file.

::: {note} Optional: Download models.
It is possible, but not required to pre-download the torch models to be used.
```sh
python -m droulette.predownload
```
This may be excessive when we only want to train one model.
:::

**Step 3.** Train models.
Finally we can train the models.

The default format trains the model in a subdirectory, to allow multiple models
of the same data.  Here we assume a subdirectory `experiment001`, containing
a configuration file `ml.toml`.  The file may look like this:

```{code-cell} ipython3
import tomllib as tl
with open( "experiment001/ml.toml", 'rb') as f:
            ml = tl.load(f)
print( json.dumps( ml, indent=4 ) )
```

To train the model, we run the command:
```sh
python -m droulette.model --config experiment001/ml.toml
```

All the results will be placed in this subdirectory.
Evaluation of these results is discussed further in [](experiment/Testing.ipynb).

::: {tip}
The droulette package also has support for batch traning multiple models
and comparing them.

This command will generate subdirectories with individual TOML files from
the given master configuration.
```sh
python -m droulette.batch experimentbatch.toml
```
Following this `droulette.model` can be run for each subdirectory.

To evaluate all the models and compare them, this command does the job:
```sh
time python -m droulette.eval -o eval.csv experiment??? 
```

This will be better documented in the future.
:::
