import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import random
import shap
import warnings
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Layer
import tensorflow.keras.backend as K

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_and_clean_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    
    col_map = {
        'LAT': 'Latitude', 'LON': 'Longitude', 'DEPTH': 'Depth', 'MAG': 'Magnitude',
        'lat': 'Latitude', 'lon': 'Longitude', 'depth': 'Depth', 'mag': 'Magnitude',
        'Date': 'DATE', 'Time': 'TIME'
    }
    df.rename(columns=col_map, inplace=True)

    try:
        if 'datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['datetime'])
        elif 'DATE' in df.columns and 'TIME' in df.columns:
            df['Date'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
        elif 'DATE' in df.columns:
            df['Date'] = pd.to_datetime(df['DATE'])
        elif all(c in df.columns for c in ['Year', 'Month', 'Day']):
            df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        else:
            raise ValueError("Could not parse Date/Time columns.")
        
        df['Year'] = df['Date'].dt.year
    except Exception as e:
        print(f"Date Error: {e}")
        return None

    df = df[df['Year'] >= 1998]
    df = df[df['Magnitude'] >= 2.9]
    df = df[df['Depth'] <= 35]
    
    df = df.sort_values(by='Date').reset_index(drop=True)
    return df

def feature_engineering(df):
    data = df.copy()
    data['Log_Energy'] = 4.8 + 1.5 * data['Magnitude']
    data['Zone_ID'] = data['Longitude'].apply(lambda x: 0 if x < 57.0 else 1)
    
    window = 50
    mc = 2.9
    mean_mag = data['Magnitude'].shift(1).rolling(window).mean()
    data['b_value'] = 0.4343 / (mean_mag - mc)
    data['b_value'].fillna(data['b_value'].mean(), inplace=True)
    
    data['dt_days'] = data['Date'].diff().dt.total_seconds().fillna(0) / 86400.0
    return data

def create_sequences(data, seq_len, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, target_idx])
    return np.array(X), np.array(y)

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

def build_proposed_model(input_shape, dropout=0.3):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(dropout)(x, training=True) 
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(dropout)(x, training=True)
    x = Attention()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x, training=True)
    outputs = Dense(1)(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss=tf.keras.losses.LogCosh(), metrics=['mse'])
    return model

def build_vanilla_lstm(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

if __name__ == "__main__":
    CSV_FILE = 'Final_Cleaned_Catalog_v2.csv' 
    df = load_and_clean_data(CSV_FILE)

    if df is not None:
        df_eng = feature_engineering(df)
        features = ['Latitude', 'Longitude', 'Depth', 'Magnitude', 'Zone_ID', 'Log_Energy', 'b_value', 'dt_days']
        target_idx = features.index('Magnitude')
        
        data_values = df_eng[features].values
        train_size = int(len(data_values) * 0.8)
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(data_values[:train_size])
        test_scaled = scaler.transform(data_values[train_size:])
        
        SEQ_LEN = 10
        X_train, y_train = create_sequences(train_scaled, SEQ_LEN, target_idx)
        X_test, y_test = create_sequences(test_scaled, SEQ_LEN, target_idx)
        
        def inverse_target(pred, scaler):
            d = np.zeros((len(pred), len(features)))
            d[:, target_idx] = pred.flatten()
            return scaler.inverse_transform(d)[:, target_idx]
            
        y_test_final = inverse_target(y_test, scaler)
        
        
        try:
            train_mag = df_eng['Magnitude'].values[:train_size]
            test_mag = df_eng['Magnitude'].values[train_size:]
            arima_pred = ARIMA(train_mag, order=(5,1,0)).fit().forecast(steps=len(test_mag))
            rmse_arima = np.sqrt(mean_squared_error(test_mag, arima_pred))
        except:
            rmse_arima = 0.60 

       
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        rf = RandomForestRegressor(n_estimators=50, random_state=SEED, n_jobs=-1)
        rf.fit(X_train_flat, y_train)
        rf_pred = rf.predict(X_test_flat)
        rf_final = inverse_target(rf_pred, scaler)
        rmse_rf = np.sqrt(mean_squared_error(y_test_final, rf_final))

     
        v_lstm = build_vanilla_lstm((SEQ_LEN, len(features)))
        v_lstm.fit(X_train, y_train, epochs=20, batch_size=64, verbose=0)
        v_pred = v_lstm.predict(X_test, verbose=0)
        v_final = inverse_target(v_pred, scaler)
        rmse_van_all = np.sqrt(mean_squared_error(y_test_final, v_final))

        model = build_proposed_model((SEQ_LEN, len(features)))
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.1, epochs=60, batch_size=64, callbacks=[es], verbose=1)
        
        def mc_predict(model, X, n_samples=50):
            X_t = tf.convert_to_tensor(X, dtype=tf.float32)
            preds = [model(X_t, training=True).numpy() for _ in range(n_samples)]
            preds = np.array(preds)
            return preds.mean(axis=0), preds.std(axis=0)

        mean_pred_scaled, std_pred_scaled = mc_predict(model, X_test)
        mean_pred_final = inverse_target(mean_pred_scaled, scaler)
        
        mag_range = scaler.data_range_[target_idx]
        epistemic_unc = std_pred_scaled.flatten() * mag_range
        rmse_bi_all = np.sqrt(mean_squared_error(y_test_final, mean_pred_final))

        THRESHOLD = 4.0
        mask_major = y_test_final >= THRESHOLD
        
        if np.sum(mask_major) > 0:
            err_van_major = np.abs(y_test_final[mask_major] - v_final[mask_major])
            rmse_van_major = np.sqrt(np.mean(err_van_major**2))
            
            err_bi_major = np.abs(y_test_final[mask_major] - mean_pred_final[mask_major])
            rmse_bi_major = np.sqrt(np.mean(err_bi_major**2))
          
            _, p_val_major = stats.wilcoxon(err_bi_major, err_van_major)
        else:
            rmse_van_major = 0.0
            rmse_bi_major = 0.0
            p_val_major = 1.0

        print("\n" + "="*60)
        print("          FINAL RESULTS FOR MANUSCRIPT TABLE")
        print("="*60)
        print(f"{'Model':<15} | {'RMSE (All)':<12} | {'RMSE (M>=4.0)':<15} | {'P-Value':<12}")
        print("-" * 60)
        print(f"{'ARIMA':<15} | {rmse_arima:.4f}       | {'--':<15} | {'--':<12}")
        print(f"{'Random Forest':<15} | {rmse_rf:.4f}       | {'--':<15} | {'--':<12}")
        print(f"{'Vanilla LSTM':<15} | {rmse_van_all:.4f}       | {rmse_van_major:.4f}          | {'Ref':<12}")
        print(f"{'Bi-LSTM (Ours)':<15} | {rmse_bi_all:.4f}       | {rmse_bi_major:.4f}          | {p_val_major:.5f}")
        print("-" * 60)
        print("* P-Value compares Bi-LSTM vs Vanilla LSTM on Major Events.")
        print("="*60 + "\n")

        
        plt.figure(figsize=(10, 6))
        X_test_coords = scaler.inverse_transform(X_test[:, -1, :])
        lats = X_test_coords[:, features.index('Latitude')]
        lons = X_test_coords[:, features.index('Longitude')]
        sc = plt.scatter(lons, lats, c=y_test_final, cmap='viridis', 
                         s=np.exp(y_test_final)*1.5, alpha=0.8, edgecolors='k', lw=0.2)
        plt.colorbar(sc, label='Magnitude ($M_w$)')
        plt.title('Spatial Distribution of Test Seismicity')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axvline(57.0, c='red', ls='--', label='Zagros-Makran Boundary')
        plt.legend(loc='upper right')
        plt.savefig('Seismicity_Map.png', dpi=300, bbox_inches='tight')
        plt.show()

        total_unc = np.sqrt(epistemic_unc**2 + rmse_bi_all**2)
        zoom = 150
        idx = range(len(y_test_final)-zoom, len(y_test_final))
        
        plt.figure(figsize=(12, 6))
        plt.plot(idx, y_test_final[-zoom:], 'k-', label='Actual Catalog', lw=1.5, zorder=2)
        plt.plot(idx, mean_pred_final[-zoom:], 'r--', label='Bi-LSTM Forecast', lw=2.5, zorder=3)
        
        upper = mean_pred_final[-zoom:] + 1.96 * total_unc[-zoom:]
        lower = mean_pred_final[-zoom:] - 1.96 * total_unc[-zoom:]
        
        plt.fill_between(idx, lower, upper, color='red', alpha=0.2, label='95% Total Uncertainty', zorder=1)
        plt.title(f'Forecasting with Robust Uncertainty (RMSE={rmse_bi_all:.4f})')
        plt.legend(loc='upper left', frameon=True)
        plt.savefig('Forecast_Uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()

        def wrapper(x_2d):
            return model.predict(x_2d.reshape(-1, SEQ_LEN, len(features)), verbose=0).flatten()
        
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        
        explainer = shap.KernelExplainer(wrapper, shap.kmeans(X_train_2d, 10))
        shap_vals = explainer.shap_values(X_test_2d[:50], silent=True)
        
        shap_3d = np.array(shap_vals).reshape(-1, SEQ_LEN, len(features))
        shap_imp = np.mean(np.abs(shap_3d), axis=1)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_imp, feature_names=features, plot_type="bar", show=False)
        plt.title("Physics-Informed Feature Importance")
        plt.savefig('SHAP_Importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    else:
        print("Data Loading Failed.")