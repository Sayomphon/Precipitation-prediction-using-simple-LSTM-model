# Precipitation Prediction using Simple LSTM

This project aims to predict daily precipitation using historical weather data. It employs a simple Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras.

## Dataset

* The data used is from the `data.csv` file, which contains daily weather observations such as temperature (average, max, min), wind speed, wind direction, and precipitation amounts.
* Data from the year 2016 is used for **training** and **validation** (split 80% for train, 20% for validation).
* Data from the year 2017 is used for **testing** the model's performance.

**Data Sample (`data.csv`):**
![Sample data.csv](image_f9bff5.png)

## Features Used

The following features were selected and preprocessed to be used as input for the LSTM model:

* `month_sin`, `month_cos`: Cyclical encoding of the month.
* `Data.Temperature.Avg Temp`: Average temperature (Scaled).
* `Data.Temperature.Max Temp`: Maximum temperature (Scaled).
* `Data.Temperature.Min Temp`: Minimum temperature (Scaled).
* `Data.Wind.Speed`: Wind speed (Scaled).

**Target Variable:**

* `Data.Precipitation`: Precipitation amount (Scaled for training, inverse-scaled for evaluation).

## Methodology / Workflow

1.  **Import Libraries:** Load necessary libraries like pandas, numpy, tensorflow, matplotlib, scikit-learn.
2.  **Load Data:** Read data from the `data.csv` file.
3.  **Preprocessing:**
    * Select relevant columns.
    * Split data into Train/Validation (2016) and Test (2017) sets.
    * Perform cyclical encoding for `Date.Month`.
    * Process `Data.Wind.Direction` (convert to numeric, cyclical encoding - *though not used as a final input feature in the provided code snippet*).
    * Scale numerical features (temperature, wind speed) and the target variable (`Data.Precipitation`) using `StandardScaler` (fit only on training data).
4.  **Data Generation:** Create time series sequences using `keras.preprocessing.timeseries_dataset_from_array` with a sequence length of 14 days (`n_steps`).
5.  **Define Model:** Build simple LSTM models with varying numbers of units (16, 32, 64, 128).
6.  **Compile Model:** Compile models using the Adam optimizer, Mean Squared Error (MSE) loss, and track Mean Absolute Error (MAE).
7.  **Train Model:** Train each model using the training data and validate with the validation data, employing `EarlyStopping` to prevent overfitting based on validation loss.
8.  **Test Model and Evaluate:**
    * Make predictions on the test set.
    * Inverse transform the predictions back to the original scale.
    * Calculate performance metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).
9.  **Plot Results:** Plot Training & Validation Loss curves and graphs comparing true vs. predicted values.

## Model Architecture

A simple LSTM architecture was used, consisting of:

1.  **Input Layer:** Expects sequences of shape `(14, 6)` (14 time steps, 6 features).
2.  **LSTM Layer:** An LSTM layer with a varying number of units (16, 32, 64, or 128).
3.  **Dense Layer:** A single output unit for predicting the precipitation value.

## Results

The performance of the LSTM models with different numbers of units was evaluated on the test set (data from 2017). Key metrics (MSE, MAE, R²) were calculated for each model configuration (refer to the notebook or script output for detailed results).

**Training & Validation Loss Plot:**

*(This graph shows the loss on the training and validation sets for each epoch for the different LSTM sizes.)*

**True vs. Predicted Precipitation Plot (Test Set):**
*(The code generates plots comparing the actual precipitation values (True) against the values predicted by the model (Predicted) for each LSTM configuration (16, 32, 64, 128) on the test set. Please refer to the notebook output for these plots.)*

## Requirements

You need to install the following Python libraries:

```bash
pip install pandas numpy tensorflow matplotlib scikit-learn