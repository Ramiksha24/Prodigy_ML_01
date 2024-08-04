# House Price Prediction

This repository contains Jupyter Notebooks for training and testing a model to predict house prices using linear regression. The dataset used is from the Kaggle House Prices competition.

## Project Description

The goal of this project is to predict house prices based on various features using a linear regression model. The project includes data preprocessing, model training, and evaluation using a test dataset.

## Files

### Training

- `train/train.csv`: Training dataset.
- `train/housepricing.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, and saving the model.

### Testing

- `test/test.csv`: Test dataset.
- `test/housepricingtest.ipynb`: Jupyter Notebook containing the code for loading the saved model, making predictions on the test data, and evaluating the predictions.

## How to Run

### Training

1. Navigate to the `train` directory.
2. Open the Jupyter Notebook `housepricing.ipynb`.
3. Execute the cells to train the model and save it.

### Testing

1. Navigate to the `test` directory.
2. Open the Jupyter Notebook `housepricingtest.ipynb`.
3. Execute the cells to load the saved model, make predictions on the test data, and evaluate the predictions.

## Requirements

The following Python libraries are required to run the notebooks:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

You can install these libraries using `pip`:
```sh
pip install pandas numpy scikit-learn matplotlib seaborn joblib
