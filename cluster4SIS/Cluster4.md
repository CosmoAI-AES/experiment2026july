
# Experiment 3 (4xSIS)

+ [](Dataset.ipynb)
+ [](experiment003/Testing.ipynb)

The datasets used in the demo are the following.

| Datasets | Ground Truth | Prediction |
| :-       | :-           | :-         |
| Training  | [training.csv](training.csv) | [training.csv](experiment003/training.csv) |
| Validation  | [validation.csv](validation.csv) | [validation.csv](experiment003/validation.csv) |
| Testing  | [testing.csv](testing.csv) | [testing.csv](experiment003/testing.csv) |

The ground truth datasets were created from the following.

+ Dataset distribution: [dataset.toml](dataset.toml)
+ Problem configuration: [problem.toml](problem.toml)
+ [dataset.csv](dataset.csv) which provides the lens parameters
  for simulation and calculation of roulette parameters.
+ [roulette.csv](roulette.csv) is the complete simulator output,
  including some columns ommitted in the data for machine learning.

+ [experiment003/training_log.csv](experiment003/training_log.csv)
  gives the evolution log from training.
  This should be commented and analysed, but this has to wait for now.

