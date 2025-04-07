# -*- coding: utf-8 -*-
"""Precipitation_prediction_using_simple_LSTM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17xXKJ48oKojAnxJZCGo1uVVYOzBGeTfJ

## **Section 1: Import libraries**
"""

!pip install tensorflow

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

"""## **Section 2: Load data**"""

data = pd.read_csv('data.csv')

data.head()

"""## **Section 3: Preprocessing**

###***Section 3.1: Select features and split data***
"""

# Select only the desired features
features_to_select = [
    'Date.Year',
    'Date.Month',
    'Data.Temperature.Avg Temp',
    'Data.Temperature.Max Temp',
    'Data.Temperature.Min Temp',
    'Data.Wind.Direction',
    'Data.Wind.Speed',
    'Data.Precipitation'
]
data_selected = data[features_to_select].copy()

# Split data into train/validation (year 2016) and test (year 2017)
train_val_data = data_selected[data_selected['Date.Year'] == 2016].copy()
test_data = data_selected[data_selected['Date.Year'] == 2017].copy()

# Drop the 'Date.Year' column (not used as a feature)
train_val_data.drop('Date.Year', axis=1, inplace=True)
test_data.drop('Date.Year', axis=1, inplace=True)

# Split train/validation: 80% for training, 20% for validation
val_split_index = int(len(train_val_data) * 0.8)
train_data = train_val_data[:val_split_index].copy()
val_data = train_val_data[val_split_index:].copy()

"""###***Section 3.2: Process wind direction and date features for cyclical encoding***"""

# Process wind direction and date features for cyclical encoding
for df in [train_data, val_data, test_data]:
    # Convert wind direction to numeric and create cyclical features for wind direction
    # Check if 'Data.Wind.Direction' column exists before processing
    if 'Data.Wind.Direction' in df.columns:
        df['Data.Wind.Direction'] = pd.to_numeric(df['Data.Wind.Direction'], errors='coerce')
        df['Data.Wind.Direction'].fillna(0, inplace=True)
        df['Data.Wind.Direction'] = df['Data.Wind.Direction'] % 360
        df['wind_dir_sin'] = np.sin(np.deg2rad(df['Data.Wind.Direction']))
        df['wind_dir_cos'] = np.cos(np.deg2rad(df['Data.Wind.Direction']))
        df.drop('Data.Wind.Direction', axis=1, inplace=True)

    # Create cyclical encoding for Date.Month
    # Check if 'Date.Month' column exists before processing
    if 'Date.Month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['Date.Month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['Date.Month'] / 12)
        # Drop the original Date.Month column as we now use cyclical features
        df.drop('Date.Month', axis=1, inplace=True)

"""###***Section 3.3: Scaling input features and target***"""

# Define the input features for the model:
# We include cyclical encoded date features, temperature features, and wind speed.
# (Note: The cyclical features 'month_sin' and 'month_cos' are already in a comparable range, so no scaling is applied)
input_features = ['month_sin', 'month_cos',
                  'Data.Temperature.Avg Temp',
                  'Data.Temperature.Max Temp',
                  'Data.Temperature.Min Temp',
                  'Data.Wind.Speed',
                  'wind_dir_sin', 'wind_dir_cos']

# Define the columns to scale (only continuous numeric features except cyclical ones)
scaling_cols = ['Data.Temperature.Avg Temp', 'Data.Temperature.Max Temp', 'Data.Temperature.Min Temp', 'Data.Wind.Speed']

# Create a scaler for the input features (fit only on train_data)
input_scaler = StandardScaler()
train_data[scaling_cols] = input_scaler.fit_transform(train_data[scaling_cols])
val_data[scaling_cols] = input_scaler.transform(val_data[scaling_cols])
test_data[scaling_cols] = input_scaler.transform(test_data[scaling_cols])

# Create a scaler for the target variable (Data.Precipitation)
target_scaler = StandardScaler()
# Create a new column for the scaled target
train_data['Data.Precipitation_scaled'] = target_scaler.fit_transform(train_data[['Data.Precipitation']])
val_data['Data.Precipitation_scaled'] = target_scaler.transform(val_data[['Data.Precipitation']])
test_data['Data.Precipitation_scaled'] = target_scaler.transform(test_data[['Data.Precipitation']])

"""## **Section 4: Create data generator**"""

# Define time series parameters
n_steps = 14
batch_size = 64

# Create data generators for training, validation, and test using the selected input features and target
train_data_gen = keras.preprocessing.timeseries_dataset_from_array(
    data=train_data[input_features].values,
    targets=train_data['Data.Precipitation_scaled'].values,
    sequence_length=n_steps,
    batch_size=batch_size,
    shuffle=True
)

val_data_gen = keras.preprocessing.timeseries_dataset_from_array(
    data=val_data[input_features].values,
    targets=val_data['Data.Precipitation_scaled'].values,
    sequence_length=n_steps,
    batch_size=batch_size,
    shuffle=False
)

test_data_gen = keras.preprocessing.timeseries_dataset_from_array(
    data=test_data[input_features].values,
    targets=test_data['Data.Precipitation_scaled'].values,
    sequence_length=n_steps,
    batch_size=batch_size,
    shuffle=False
)

"""## **Section 5: Define model**"""

# The number of features used in the generator is now 6
num_features = len(input_features)

# Define number of EarlyStopping patiene
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

"""### ***Section 5.1: Model with LSTM units: 16***"""

model16 = keras.Sequential([
    keras.layers.Input(shape=(n_steps, num_features)),
    keras.layers.LSTM(16),
    keras.layers.Dense(1)
])
model16.summary()

"""### ***Section 5.2: Model with LSTM units: 32***"""

model32 = keras.Sequential([
    keras.layers.Input(shape=(n_steps, num_features)),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])
model32.summary()

"""### ***Section 5.3: Model with LSTM units: 64***"""

model64 = keras.Sequential([
    keras.layers.Input(shape=(n_steps, num_features)),
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])
model64.summary()

"""### ***Section 5.4: Model with LSTM units: 128***"""

model128 = keras.Sequential([
    keras.layers.Input(shape=(n_steps, num_features)),
    keras.layers.LSTM(128),
    keras.layers.Dense(1)
])
model128.summary()

"""## **Section 6: Compile model**"""

# Define Learning rate
learning_rate=0.0001

# Compile the models with the same configuration:
# Model with 16 LSTM units
model16.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mae'])

# Model with 32 LSTM units
model32.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mae'])

# Model with 64 LSTM units
model64.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mae'])

# Model with 128 LSTM units
model128.compile(loss='mse',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['mae'])

"""## **Section 7: Train model**"""

# Define model training parameters
epochs=50
callbacks=[early_stopping]

"""### ***Section 7.1: Train the model with 16 LSTM units***"""

# Train the model with 16 LSTM units:
history16 = model16.fit(train_data_gen,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        callbacks=callbacks,
                        verbose=1)

"""### ***Section 7.2: Train the model with 32 LSTM units***"""

# Train the model with 32 LSTM units:
history32 = model32.fit(train_data_gen,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        callbacks=callbacks,
                        verbose=1)

"""### ***Section 7.3: Train the model with 64 LSTM units***"""

# Train the model with 64 LSTM units:
history64 = model64.fit(train_data_gen,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        callbacks=callbacks,
                        verbose=1)

"""### ***Section 7.4: Train the model with 128 LSTM units***"""

# Train the model with 128 LSTM units:
history128 = model128.fit(train_data_gen,
                        epochs=epochs,
                        validation_data=val_data_gen,
                        callbacks=callbacks,
                        verbose=1)

"""## **Section 8: Test model and print results**"""

# Create test samples manually from test_data (ใช้ input_features เดียวกันสำหรับทุกโมเดล)
test_samples = len(test_data) - n_steps
X_test = np.array([
    test_data[input_features].values[i:i+n_steps]
    for i in range(test_samples)
])
# Get true target values (scaled) and then convert to original scale
y_true_scaled = test_data['Data.Precipitation_scaled'].values[n_steps:].reshape(-1, 1)
y_true = target_scaler.inverse_transform(y_true_scaled)

"""### ***Section 8.1: Evaluate Model 16***"""

# Predict on the test set using the trained model (model16)
y_pred16_scaled = model16.predict(X_test)

# Inverse transform the scaled predictions to get the predictions in the original scale
y_pred16 = target_scaler.inverse_transform(y_pred16_scaled)

# Calculate the Mean Squared Error (MSE), Mean Absolute Error (MAE) and R-squared (R²) score
mse16 = MeanSquaredError()(y_true, y_pred16).numpy()
mae16 = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred16).numpy()
r2_16 = r2_score(y_true, y_pred16)

print("LSTM 16 - MSE: ", mse16, "MAE: ", mae16, "R²: ", r2_16)

"""### ***Section 8.2: Evaluate Model 32***"""

# Predict on the test set using the trained model (model32)
y_pred32_scaled = model32.predict(X_test)

# Inverse transform the scaled predictions to get the predictions in the original scale
y_pred32 = target_scaler.inverse_transform(y_pred32_scaled)

# Calculate the Mean Squared Error (MSE), Mean Absolute Error (MAE) and R-squared (R²) score
mse32 = MeanSquaredError()(y_true, y_pred32).numpy()
mae32 = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred32).numpy()
r2_32 = r2_score(y_true, y_pred32)

print("LSTM 32 - MSE: ", mse32, "MAE: ", mae32, "R²: ", r2_32)

"""### ***Section 8.3: Evaluate Model 64***"""

# Predict on the test set using the trained model (model64)
y_pred64_scaled = model64.predict(X_test)

# Inverse transform the scaled predictions to get the predictions in the original scale
y_pred64 = target_scaler.inverse_transform(y_pred64_scaled)

# Calculate the Mean Squared Error (MSE), Mean Absolute Error (MAE) and R-squared (R²) score
mse64 = MeanSquaredError()(y_true, y_pred64).numpy()
mae64 = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred64).numpy()
r2_64 = r2_score(y_true, y_pred64)

print("LSTM 64 - MSE: ", mse64, "MAE: ", mae64, "R²: ", r2_64)

"""### ***Section 8.4: Evaluate Model 128***"""

# Predict on the test set using the trained model (model128)
y_pred128_scaled = model128.predict(X_test)

# Inverse transform the scaled predictions to get the predictions in the original scale
y_pred128 = target_scaler.inverse_transform(y_pred128_scaled)

# Calculate the Mean Squared Error (MSE), Mean Absolute Error (MAE) and R-squared (R²) score
mse128 = MeanSquaredError()(y_true, y_pred128).numpy()
mae128 = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred128).numpy()
r2_128 = r2_score(y_true, y_pred128)

print("LSTM 128 - MSE: ", mse128, "MAE: ", mae128, "R²: ", r2_128)

"""## **Section 9: Plot true vs predicted precipitation values**

###***Section 9.1: Plot Training & Validation Loss for Each Mode***
"""

plt.figure(figsize=(12, 8))
# Plot for model 16
epochs16 = range(1, len(history16.history['loss']) + 1)
plt.plot(epochs16, history16.history['loss'], label='Train Loss (LSTM 16)')
plt.plot(epochs16, history16.history['val_loss'], '--', label='Val Loss (LSTM 16)')

# Plot for model 32
epochs32 = range(1, len(history32.history['loss']) + 1)
plt.plot(epochs32, history32.history['loss'], label='Train Loss (LSTM 32)')
plt.plot(epochs32, history32.history['val_loss'], '--', label='Val Loss (LSTM 32)')

# Plot for model 64
epochs64 = range(1, len(history64.history['loss']) + 1)
plt.plot(epochs64, history64.history['loss'], label='Train Loss (LSTM 64)')
plt.plot(epochs64, history64.history['val_loss'], '--', label='Val Loss (LSTM 64)')

# Plot for model 128
epochs128 = range(1, len(history128.history['loss']) + 1)
plt.plot(epochs128, history128.history['loss'], label='Train Loss (LSTM 128)')
plt.plot(epochs128, history128.history['val_loss'], '--', label='Val Loss (LSTM 128)')

plt.title('Training and Validation Loss for Different LSTM Units')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

"""###***Section 9.2: Plot True vs Predicted Precipitation for Each Model***"""

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(y_true, label='True')
plt.plot(y_pred16, label='Predicted')
plt.title('LSTM 16')
plt.xlabel('Sample index (Test Set)')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(y_true, label='True')
plt.plot(y_pred32, label='Predicted')
plt.title('LSTM 32')
plt.xlabel('Sample index (Test Set)')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(y_true, label='True')
plt.plot(y_pred64, label='Predicted')
plt.title('LSTM 64')
plt.xlabel('Sample index (Test Set)')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(y_true, label='True')
plt.plot(y_pred128, label='Predicted')
plt.title('LSTM 128')
plt.xlabel('Sample index (Test Set)')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()