
# Experiment 1 (singleton SIE)

+ [](Dataset.ipynb)
+ [](experiment001/Testing.ipynb)

The datasets used in the demo are the following.

| Datasets | Ground Truth | Prediction |
| :-       | :-           | :-         |
| Training  | [sie-training.csv](sie-training.csv) | [pred-sie-train.csv](experiment001/pred-sie-train.csv) |
| Validation  | [sie-validation.csv](sie-validation.csv) | [pred-sie-val.csv](experiment001/pred-sie-val.csv) |
| Testing  | [sie-testing.csv](sie-testing.csv) | [pred-sie-test.csv](experiment001/pred-sie-test.csv) |

The ground truth datasets were created from the following.

+ Dataset distribution: [sie-dataset.toml](sie-dataset.toml)
+ Problem configuration: [sie-problem.toml](sie-problem.toml)
+ [sie-dataset.csv](sie-dataset.csv) which provides the lens parameters
  for simulation and calcuation of roulette parameters.
+ [sie-roulette.csv](sie-roulette.csv) is the complete simulator output,
  including some columns ommitted in the data for machine learning.

+ [experiment001/training_log.csv](experiment001/training_log.csv)
  gives the evolution log from training.
  This should be commented and analysed, but this has to wait for now.

