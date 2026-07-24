# CosmoSim Experiments July 2026

These pages report on three experiments made July 2026.
+ [](SIE/SIE.md)
+ [](cluster2SIE/Cluster2.md)
+ [](cluster4SIS/Cluster4.md)

All the experiments follow the same general protocol, elaborate
in [](Training.ipynb), with the following steps.
1. Dataset generation.
2. Training of a single machine learning model.
3. Testing and analysis.

This is based on the general [](xref:cosmoai/pipeline#ml-pipeline).

You can download and run the different documents in Jupyter Lab.
Notebooks and datafiles can be downloaded one by one from the pages,
or you can clone the entire site from
[github](https://github.com/CosmoAI-AES/experiment2026july/).

To install the dependencies, download 
[requirements.txt](requirements.txt) and use
```sh
pip install -r requirements.txt
```

To open the files from git in jupyter lab, they must be converted
from md:myst to ipynb by `jupytext`.
If you have `make` you can use the Makefile provided.
```sh
make ipynb
```

::: {note} Software Versions
The experiments are executed with CosmoSim v3.3 and droulette v0.2.1.

CosmoSim v3.2.4 gives a numeric inaccuracy in the resimulation
from roulette amplitudes, although other parts of the experiment
will work.

Earlier versions are not expected to work, for various reasons.
:::


**Home** to [](xref:cosmoai)

