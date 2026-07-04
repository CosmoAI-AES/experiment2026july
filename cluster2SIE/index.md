
# Experiment 2 (2xSIE)

+ [](Dataset.ipynb)
+ [](experiment002/Testing.ipynb)
+ See also [](/docs/ML/Pipeline).

The datasets used in the demo are the following.

| Datasets | Ground Truth | Prediction |
| :-       | :-           | :-         |
| Training  | [training.csv](training.csv) | [train.csv](experiment002/train.csv) |
| Validation  | [validation.csv](validation.csv) | [val.csv](experiment002/val.csv) |
| Testing  | [testing.csv](testing.csv) | [test.csv](experiment002/test.csv) |

The ground truth datasets were created from the following.

+ Dataset distribution: [dataset.toml](dataset.toml)
+ Problem configuration: [problem.toml](problem.toml)
+ [dataset.csv](dataset.csv) which provides the lens parameters
  for simulation and calcuation of roulette parameters.
+ [roulette.csv](roulette.csv) is the complete simulator output,
  including some columns ommitted in the data for machine learning.

+ [experiment002/training_log.csv](experiment002/training_log.csv)
  gives the evolution log from training.
  This should be commented and analysed, but this has to wait for now.


