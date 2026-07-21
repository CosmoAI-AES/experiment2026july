
# Experiment 2 (2xSIE)

+ [](Dataset.ipynb)
+ [](experiment002/Testing.ipynb)

The datasets used in the demo are the following.

| Datasets | Ground Truth | Prediction |
| :-       | :-           | :-         |
| Training  | [training.csv](training.csv) | [cluster2-training.csv](experiment002/cluster2-training.csv) |
| Validation  | [validation.csv](validation.csv) | [cluster2-validation.csv](experiment002/cluster2-validation.csv) |
| Testing  | [testing.csv](testing.csv) | [cluster2-testing.csv](experiment002/cluster2-testing.csv) |

The ground truth datasets were created from the following.

+ Dataset distribution: [cluster2-dataset.toml](cluster2-dataset.toml)
+ Problem configuration: [cluster2-problem.toml](cluster2-problem.toml)
+ [cluster2-dataset.csv](cluster2-dataset.csv) which provides the lens parameters
  for simulation and calculation of roulette parameters.
+ [cluster2-roulette.csv](cluster2-roulette.csv) is the complete simulator output,
  including some columns ommitted in the data for machine learning.

+ [experiment002/training_log.csv](experiment002/training_log.csv)
  gives the evolution log from training.
  This should be commented and analysed, but this has to wait for now.


