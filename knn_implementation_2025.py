path = '/Users/jacob/Documents/M.S. COMPUTER SCIENCE/Spring 2023 - CPEG589, CPSC552, CPSC501/Data Mining/Assignment 4 - K-Nearest Neighbors'

# KNN can be used for Classification or Regression
### Has one hyperparameter (k) specifying num_neighbors

# GOAL: Run regression to predict age of the abalone ("Rings") based on
# eight features (X1, ... X8)
### Experiment with (k), num_neighbors, to determine highest accuracy

### UTILITIES (Utils.py)

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import *

# Retrieve dataset from UCI .data url / Return as pandas df
def get_dataset():
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    url = (
"https://archive.ics.uci.edu/ml/machine-learning-databases"
"/abalone/abalone.data"
)
    abalone_df = pd.read_csv(url, header=None)
    abalone_df.columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    return abalone_df

    # Gave error: Python unable to verify SSL certificate of website
    # Resolved with terminal command: '/Applications/Python\ 3.12/Install\ Certificates.command'

def get_train_test_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)
    return x_train, x_test, y_train, y_test

def plot_predicted_vs_actual(y_pred, y):
    mean_error = sum(abs(y_pred-y))/len(y)
    step_size = 10
    a = [y_pred[i] for i in range(0, len(y_pred)) if i%step_size==0]
    b = [y[i] for i in range(0, len(y_pred)) if i%step_size==0]
    t = linspace(0, len(a), len(a))

    plt.plot(t, a, 'red', linestyle='dashed', label='predicted')
    plt.plot(t, b, 'blue', label='actual')
    plt.scatter(t, a, marker='o', s=10, color='red', label='predicted')
    plt.scatter(t, b, s=10, color='blue', label='actual')

    plt.legend()
    plt.title('Mean Error ='+ str(mean_error))
    plt.show()

import sys
import scipy.stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def main():
    df = get_dataset()
    # print(df.head)

    df = df.drop("Sex", axis=1)

    # df["Rings"].hist(bins=15)
    # plt.show()

    corr_matrix = df.corr()
    pd.reset_option('display.max_columns')
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.width', None)
    print(corr_matrix["Rings"].round(2))

    X_df = df.drop("Rings", axis=1) # Drop target variable
    print(type(X_df)) # <class 'pandas.core.frame.DataFrame'>
    X = X_df.values # Convert to numpy
    print(type(X)) # <class 'numpy.ndarray'>
    print(X.shape)

    y_df = df["Rings"]
    y = y_df.values
    print(type(y), y.shape)


    ################################
    # Test KNN on unknown data point
    ################################

    new_data_point = np.array([
        0.417, # length
        0.396, # diameter
        0.134, # height
        0.816, # whole weight
        0.383, # shucked weight
        0.172, # viscera weight
        0.221])

    distances = np.linalg.norm(X - new_data_point, axis=1)
    k = 15

    nearest_neighbor_ids = distances.argsort()[:k] # Top k neighbors' indices
    print(nearest_neighbor_ids)

    nearest_neighbor_rings = y[nearest_neighbor_ids]
    print(nearest_neighbor_rings)

    # USE MEAN FOR PREDICTION
    prediction = nearest_neighbor_rings.mean()
    print(f'Predicted rings using mean: {prediction.round(1)}')

    mode = scipy.stats.mode(nearest_neighbor_rings)
    print('Predicted rings using mode:', mode[0])

    ################################
    # Use KNN with SCIKIT-LEARN
    ################################

    x_train, x_test, y_train, y_test = get_train_test_data(X, y)
    knn_model = KNeighborsRegressor(n_neighbors=8)
    knn_model.fit(x_train, y_train)

    train_predictions = knn_model.predict(x_train)
    mse = mean_squared_error(y_train, train_predictions)
    rmse = np.sqrt(mse)
    print('training error = ',rmse)

    test_preds = knn_model.predict(x_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print('testing error = ',rmse)
    plot_predicted_vs_actual(test_preds,y_test)
    # try different values of n_neighbors to see if error drops


if __name__ == "__main__":
    sys.exit(int(main() or 0))