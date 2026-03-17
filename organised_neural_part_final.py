import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Input, TimeDistributed, LSTM, GRU

df_counter = pd.read_excel("countercurrent_results.xlsx").sample(frac=1, random_state=42).reset_index(drop=True)
df_cross = pd.read_excel("crosscurrent_results.xlsx").sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCountercurrent Data Overview:")
print(f"Total Samples: {len(df_counter)}")
print(df_counter[['Feed Flowrate', 'Solvent Flowrate', 'xcf']].describe())

print("\nCrosscurrent Data Overview:")
print(f"Total Samples: {len(df_cross)}")
print(df_cross[['Feed Flowrate', 'Solvent Flowrate', 'xcf']].describe())

# In an extraction column, the physical stages act sequentially.
# The mass transfer at Stage N heavily depends on Stage N-1.
num_stages = 20
N_counter = len(df_counter)
N_cross = len(df_cross)

# Initialize arrays for RNN: Shape (Samples, 20 Time Steps, 4 Features)
Xc_rnn = np.zeros((N_counter, num_stages, 4))
yc_rnn = np.zeros((N_counter, num_stages, 1))

Xx_rnn = np.zeros((N_cross, num_stages, 4))
yx_rnn = np.zeros((N_cross, num_stages, 1))

# Populate Countercurrent Arrays
for i in range(N_counter):
    row = df_counter.iloc[i]
    feed, solvent, xcf = row['Feed Flowrate'], row['Solvent Flowrate'], row['xcf']
    for stage in range(1, num_stages + 1):
        Xc_rnn[i, stage-1] = [feed, solvent, xcf, stage]
        yc_rnn[i, stage-1, 0] = row[f'Stage_{stage}']

# Populate Crosscurrent Arrays
for i in range(N_cross):
    row = df_cross.iloc[i]
    feed, solvent, xcf = row['Feed Flowrate'], row['Solvent Flowrate'], row['xcf']
    for stage in range(1, num_stages + 1):
        Xx_rnn[i, stage-1] = [feed, solvent, xcf, stage]
        yx_rnn[i, stage-1, 0] = row[f'Stage_{stage}']

# Flatten the sequential data for the ANN
# The ANN will treat every single stage as a completely independent flash drum,
# ignoring the upstream/downstream connections.
Xc_ann = Xc_rnn.reshape(-1, 4)
yc_ann = yc_rnn.reshape(-1, 1)

Xx_ann = Xx_rnn.reshape(-1, 4)
yx_ann = yx_rnn.reshape(-1, 1)

scaler_c = StandardScaler().fit(Xc_ann)
Xc_ann_scaled = scaler_c.transform(Xc_ann)
Xc_rnn_scaled = Xc_ann_scaled.reshape(-1, num_stages, 4)

scaler_x = StandardScaler().fit(Xx_ann)
Xx_ann_scaled = scaler_x.transform(Xx_ann)
Xx_rnn_scaled = Xx_ann_scaled.reshape(-1, num_stages, 4)

# Split flat data for ANN
Xc_a_train, Xc_a_test, yc_a_train, yc_a_test = train_test_split(Xc_ann_scaled, yc_ann, test_size=0.2, random_state=42)
Xx_a_train, Xx_a_test, yx_a_train, yx_a_test = train_test_split(Xx_ann_scaled, yx_ann, test_size=0.2, random_state=42)

# Split sequential data for RNN (keeps the 20-stage blocks intact)
Xc_r_train, Xc_r_test, yc_r_train, yc_r_test = train_test_split(Xc_rnn_scaled, yc_rnn, test_size=0.2, random_state=42)
Xx_r_train, Xx_r_test, yx_r_train, yx_r_test = train_test_split(Xx_rnn_scaled, yx_rnn, test_size=0.2, random_state=42)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda

print("\n--- Training Countercurrent Models ---")

# --- ANN ---
ann_c = Sequential([
    Input(shape=(4,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
    Lambda(lambda x: x * 100)
])
ann_c.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training Countercurrent ANN...")
hist_ann_c = ann_c.fit(Xc_a_train, yc_a_train, epochs=150, batch_size=32, validation_split=0.2, verbose=0)
_, c_ann_mae = ann_c.evaluate(Xc_a_test, yc_a_test, verbose=0)
print(f"Countercurrent ANN Final Test MAE: {c_ann_mae:.4f}")

# --- RNN ---
early_stop=EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
rnn_c = Sequential([
    Input(shape=(20, 4)),
    SimpleRNN(64, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid')),
    TimeDistributed(Lambda(lambda x: x * 100))
])
rnn_c.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training Countercurrent RNN...")
hist_rnn_c = rnn_c.fit(Xc_r_train, yc_r_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
_, c_rnn_mae = rnn_c.evaluate(Xc_r_test, yc_r_test, verbose=0)
print(f"Countercurrent RNN Final Test MAE: {c_rnn_mae:.4f}")

# Plot Countercurrent Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_ann_c.history['mae'], label='Train MAE')
plt.plot(hist_ann_c.history['val_mae'], label='Val MAE')
plt.title('Countercurrent - ANN Learning Curve')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_rnn_c.history['mae'], label='Train MAE')
plt.plot(hist_rnn_c.history['val_mae'], label='Val MAE')
plt.title('Countercurrent - RNN Learning Curve')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Training Crosscurrent Models ---")

# --- ANN ---
ann_x = Sequential([
    Input(shape=(4,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid'),
    Lambda(lambda x: x * 100)
])
ann_x.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training Crosscurrent ANN...")
hist_ann_x = ann_x.fit(Xx_a_train, yx_a_train, epochs=150, batch_size=32, validation_split=0.2, verbose=0)
_, x_ann_mae = ann_x.evaluate(Xx_a_test, yx_a_test, verbose=0)
print(f"Crosscurrent ANN Final Test MAE: {x_ann_mae:.4f}")

# --- RNN ---
early_stop=EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
rnn_x = Sequential([
    Input(shape=(20, 4)),
    SimpleRNN(64, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid')),
    TimeDistributed(Lambda(lambda x: x * 100))
])
rnn_x.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training Crosscurrent RNN...")
hist_rnn_x = rnn_x.fit(Xx_r_train, yx_r_train, epochs=300, batch_size=32, callbacks=[early_stop], validation_split=0.2, verbose=0)
_, x_rnn_mae = rnn_x.evaluate(Xx_r_test, yx_r_test, verbose=0)
print(f"Crosscurrent RNN Final Test MAE: {x_rnn_mae:.4f}")

# Plot Crosscurrent Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_ann_x.history['mae'], label='Train MAE')
plt.plot(hist_ann_x.history['val_mae'], label='Val MAE')
plt.title('Crosscurrent - ANN Learning Curve')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_rnn_x.history['mae'], label='Train MAE')
plt.plot(hist_rnn_x.history['val_mae'], label='Val MAE')
plt.title('Crosscurrent - RNN Learning Curve')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Training Countercurrent Advanced Models ---")



# --- LSTM ---
# LSTMs have a separate 'cell state' that acts like a conveyor belt,
# carrying the feed and initial solvent conditions down the entire 20-stage cascade.
lstm_c = Sequential([
    Input(shape=(20, 4)),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid')),
    TimeDistributed(Lambda(lambda x: x * 100))
])
early_stop=EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
lstm_c.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training Countercurrent LSTM...")
hist_lstm_c = lstm_c.fit(Xc_r_train, yc_r_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
_, c_lstm_mae = lstm_c.evaluate(Xc_r_test, yc_r_test, verbose=0)
print(f"Countercurrent LSTM Final Test MAE: {c_lstm_mae:.4f}")



# --- GRU ---
# GRUs are a streamlined version of LSTMs. They merge the cell state and hidden state,
# making them faster to train while still retaining long-term stage dependencies.
gru_c = Sequential([
    Input(shape=(20, 4)),
    GRU(64, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid')),
    TimeDistributed(Lambda(lambda x: x * 100))
])
early_stop=EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
gru_c.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training Countercurrent GRU...")
hist_gru_c = gru_c.fit(Xc_r_train, yc_r_train, epochs=300, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
_, c_gru_mae = gru_c.evaluate(Xc_r_test, yc_r_test, verbose=0)
print(f"Countercurrent GRU Final Test MAE: {c_gru_mae:.4f}")

# Plot Countercurrent Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_lstm_c.history['mae'], label='Train MAE')
plt.plot(hist_lstm_c.history['val_mae'], label='Val MAE')
plt.title('Countercurrent - LSTM Learning Curve')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_gru_c.history['mae'], label='Train MAE')
plt.plot(hist_gru_c.history['val_mae'], label='Val MAE')
plt.title('Countercurrent - GRU Learning Curve')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Training Crosscurrent Advanced Models ---")

# --- LSTM ---
lstm_x = Sequential([
    Input(shape=(20, 4)),
    LSTM(64, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid')),
    TimeDistributed(Lambda(lambda x: x * 100))
])
lstm_x.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop=EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
print("Training Crosscurrent LSTM...")
hist_lstm_x = lstm_x.fit(Xx_r_train, yx_r_train, epochs=300, batch_size=32, callbacks=[early_stop], validation_split=0.2, verbose=0)
_, x_lstm_mae = lstm_x.evaluate(Xx_r_test, yx_r_test, verbose=0)
print(f"Crosscurrent LSTM Final Test MAE: {x_lstm_mae:.4f}")

# --- GRU ---
gru_x = Sequential([
    Input(shape=(20, 4)),
    GRU(64, return_sequences=True),
    TimeDistributed(Dense(1, activation='sigmoid')),
    TimeDistributed(Lambda(lambda x: x * 100))
])
gru_x.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop=EarlyStopping(monitor='val_mae', patience=15, restore_best_weights=True)
print("Training Crosscurrent GRU...")
hist_gru_x = gru_x.fit(Xx_r_train, yx_r_train, epochs=300, batch_size=32, callbacks=[early_stop], validation_split=0.2, verbose=0)
_, x_gru_mae = gru_x.evaluate(Xx_r_test, yx_r_test, verbose=0)
print(f"Crosscurrent GRU Final Test MAE: {x_gru_mae:.4f}")

# Plot Crosscurrent Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(hist_lstm_x.history['mae'], label='Train MAE')
plt.plot(hist_lstm_x.history['val_mae'], label='Val MAE')
plt.title('Crosscurrent - LSTM Learning Curve')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_gru_x.history['mae'], label='Train MAE')
plt.plot(hist_gru_x.history['val_mae'], label='Val MAE')
plt.title('Crosscurrent - GRU Learning Curve')
plt.xlabel('Epochs')
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Final MAE Comparison (Lower is Better) ---")
print("Countercurrent Process:")
print(f"  SimpleRNN: {c_rnn_mae:.4f}")
print(f"  LSTM:      {c_lstm_mae:.4f}")
print(f"  GRU:       {c_gru_mae:.4f}")

print("\nCrosscurrent Process:")
print(f"  SimpleRNN: {x_rnn_mae:.4f}")
print(f"  LSTM:      {x_lstm_mae:.4f}")
print(f"  GRU:       {x_gru_mae:.4f}")

import os

# Create a directory to store models if it doesn't exist
if not os.path.exists('extraction_models'):
    os.makedirs('extraction_models')

# Dictionary mapping names to your model objects
models_to_save = {
    "ann_countercurrent": ann_c,
    "rnn_countercurrent": rnn_c,
    "lstm_countercurrent": lstm_c,
    "gru_countercurrent": gru_c,
    "ann_crosscurrent": ann_x,
    "rnn_crosscurrent": rnn_x,
    "lstm_crosscurrent": lstm_x,
    "gru_crosscurrent": gru_x
}

import joblib

# Save the scalers so the GUI can use them!
joblib.dump(scaler_c, 'extraction_models/scaler_counter.pkl')
joblib.dump(scaler_x, 'extraction_models/scaler_cross.pkl')

print("Saved scalers to extraction_models folder. Ready for GUI!")

# Loop through and save each model in the latest .keras format
for name, model in models_to_save.items():
    filepath = f'extraction_models/{name}.keras'
    model.save(filepath)
    print(f"Saved {name} to {filepath}")

import numpy as np
import matplotlib.pyplot as plt

def generate_contour_plot(model, scaler, df, model_name):
    # Fix Feed and xcf to median baseline values
    feed_baseline = df['Feed Flowrate'].median()
    xcf_baseline = df['xcf'].median()

    # Create the grid axes
    solvent_range = np.linspace(df['Solvent Flowrate'].min(), df['Solvent Flowrate'].max(), 50)
    stage_range = np.arange(1, 21)

    Z_output = np.zeros((len(solvent_range), len(stage_range)))
    is_ann = 'ann' in model_name.lower()

    if is_ann:
        for i, solvent in enumerate(solvent_range):
            for j, stage in enumerate(stage_range):
                X_in = np.array([[feed_baseline, solvent, xcf_baseline, stage]])
                X_scaled = scaler.transform(X_in)

                # Directly use the model's output (already 0-100)
                Z_output[i, j] = model.predict(X_scaled, verbose=0)[0][0]

    else: # RNN / LSTM / GRU
        for i, solvent in enumerate(solvent_range):
            X_seq = np.zeros((1, 20, 4))
            for st in range(20):
                X_seq[0, st] = [feed_baseline, solvent, xcf_baseline, st + 1]

            X_seq_flat = X_seq.reshape(-1, 4)
            X_seq_scaled = scaler.transform(X_seq_flat).reshape(1, 20, 4)
            preds = model.predict(X_seq_scaled, verbose=0)

            for j, stage in enumerate(stage_range):
                # Directly use the model's output (already 0-100)
                Z_output[i, j] = preds[0, j, 0]

    # Plotting
    plt.figure(figsize=(8, 6))
    Stage_Grid, Solvent_Grid = np.meshgrid(stage_range, solvent_range)

    # Levels set to expected model output range
    cp = plt.contourf(Stage_Grid, Solvent_Grid, Z_output, levels=np.linspace(0, 100, 20), cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.set_label('Model Output (Stage Value)')

    plt.xlabel('Number of Stages')
    plt.ylabel('Solvent Flowrate')
    plt.title(f'Extraction Optimization Contour\nModel: {model_name}')
    plt.grid(color='white', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

for name, model in models_to_save.items():
    current_scaler = scaler_c if 'countercurrent' in name else scaler_x
    current_df = df_counter if 'countercurrent' in name else df_cross
    print(f"Generating Contour for: {name}...")
    generate_contour_plot(model, current_scaler, current_df, name)

def plot_actual_vs_predicted_removal(model, X_test_scaled, y_test, scaler, model_name):
    is_ann = 'ann' in model_name.lower()

    if is_ann:
        # Get unscaled data just to find the Stage 20 indices
        X_unscaled = scaler.inverse_transform(X_test_scaled)
        stage_20_idx = np.where(np.isclose(X_unscaled[:, 3], 20))[0]

        X_test_20 = X_test_scaled[stage_20_idx]

        # Actuals and Predictions are used exactly as they are
        actual_vals = y_test[stage_20_idx].flatten()
        pred_vals = model.predict(X_test_20, verbose=0).flatten()

    else: # RNN / LSTM / GRU
        y_pred = model.predict(X_test_scaled, verbose=0)

        # Extract the final stage (index 19)
        actual_vals = y_test[:, 19, 0]
        pred_vals = y_pred[:, 19, 0]

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_vals, pred_vals, alpha=0.6, edgecolor='k', label='Test Data')

    # Create the ideal parity line based on the data's true min/max
    min_val = min(np.min(actual_vals), np.min(pred_vals))
    max_val = max(np.max(actual_vals), np.max(pred_vals))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.xlabel('Actual Target Value (Stage 20)')
    plt.ylabel('Predicted Target Value (Stage 20)')
    plt.title(f'Parity Plot: Actual vs Predicted\nModel: {model_name}')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

print("\n--- Generating Parity Plots ---")
for name, model in models_to_save.items():
    print(f"Plotting Parity for: {name}...")
    if 'ann' in name:
        X_test = Xc_a_test if 'countercurrent' in name else Xx_a_test
        y_test = yc_a_test if 'countercurrent' in name else yx_a_test
    else:
        X_test = Xc_r_test if 'countercurrent' in name else Xx_r_test
        y_test = yc_r_test if 'countercurrent' in name else yx_r_test

    current_scaler = scaler_c if 'countercurrent' in name else scaler_x
    plot_actual_vs_predicted_removal(model, X_test, y_test, current_scaler, name)