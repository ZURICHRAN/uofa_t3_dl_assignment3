import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K


# 自定义RMSE指标（可选）
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# ============== 数据加载与基本预处理 ==============
train_file_path = "Google_Stock_Price_Train.csv"
test_file_path = "Google_Stock_Price_Test.csv"
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 数据清理与转换
for data in [train_data, test_data]:
    data['Volume'] = data['Volume'].astype(str).str.replace(',', '').astype(float)
    data['Close'] = data['Close'].astype(str).str.replace(',', '').astype(float)
    data['Date'] = pd.to_datetime(data['Date'])

# 以日期为索引，这样后续绘图就能直接使用日期轴
train_data = train_data.set_index('Date').sort_index()
test_data = test_data.set_index('Date').sort_index()

# 划分数据集：使用训练集末尾30天为验证集
val_days = 30
train_data_final = train_data.iloc[:-val_days]
val_data = train_data.iloc[-val_days:]

# features = ['Open', 'High', 'Low', 'Close', 'Volume']
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data_final[features])
val_scaled = scaler.transform(val_data[features])
test_scaled = scaler.transform(test_data[features])

train_data_scaled = pd.DataFrame(train_scaled, columns=features, index=train_data_final.index)
val_data_scaled = pd.DataFrame(val_scaled, columns=features, index=val_data.index)
test_data_scaled = pd.DataFrame(test_scaled, columns=features, index=test_data.index)


# ============== 构造序列数据函数(增加dates返回值) ==============
def create_sequences(data, N, features, target):
    X = []
    y = []
    dates = []
    for i in range(len(data) - N):
        X_seq = data[features].values[i:i + N]
        y_seq = data[target].values[i + N]
        X.append(X_seq)
        y.append(y_seq)
        # 保存目标日对应的日期索引
        dates.append(data.index[i + N])
    return np.array(X), np.array(y), np.array(dates)


N = 30
X_train, y_train, train_dates = create_sequences(train_data_scaled, N, features, target)

# 验证集序列构造（方法A：拼接训练集尾部N天和验证集数据）
combined_val_data = pd.concat([train_data_scaled.iloc[-N:], val_data_scaled], axis=0)
X_val_full, y_val_full, val_dates_full = create_sequences(combined_val_data, N, features, target)
valid_indices = []
for i in range(len(combined_val_data) - N):
    target_day = i + N
    if target_day >= N:
        valid_indices.append(i)

X_val = X_val_full[valid_indices]
y_val = y_val_full[valid_indices]
val_dates = val_dates_full[valid_indices]

# 测试集序列构造（拼接验证集尾部N天和测试集数据）
combined_test_data = pd.concat([val_data_scaled.iloc[-N:], test_data_scaled], axis=0)
X_test_full, y_test_full, test_dates_full = create_sequences(combined_test_data, N, features, target)
test_valid_indices = []
for i in range(len(combined_test_data) - N):
    target_day = i + N
    if target_day >= N:
        test_valid_indices.append(i)

X_test = X_test_full[test_valid_indices]
y_test = y_test_full[test_valid_indices]
test_dates = test_dates_full[test_valid_indices]

# ============== 模型构建与训练 ==============
model = Sequential([
    Input(shape=(N, len(features))),  # 定义输入形状
    LSTM(128, return_sequences=True),
    LSTM(128, return_sequences=False),
    Dropout(0.4),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=[rmse])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint],
    verbose=1
)

# 查看训练过程
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# ============== 预测与逆归一化 ==============
test_loss = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", test_loss)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)


def inverse_transform_close(data_scaled, scaler, features, target):
    data_scaled = data_scaled.reshape(-1)
    dummy = np.zeros((len(data_scaled), len(features)))
    close_index = features.index(target)
    dummy[:, close_index] = data_scaled
    data_original = scaler.inverse_transform(dummy)
    return data_original[:, close_index]


y_val_original = inverse_transform_close(y_val, scaler, features, target)
y_val_pred_original = inverse_transform_close(y_val_pred, scaler, features, target)
y_test_original = inverse_transform_close(y_test, scaler, features, target)
y_test_pred_original = inverse_transform_close(y_test_pred, scaler, features, target)

# 将预测结果转化为带日期索引的Series
val_series_true = pd.Series(y_val_original, index=val_dates)
val_series_pred = pd.Series(y_val_pred_original, index=val_dates)

test_series_true = pd.Series(y_test_original, index=test_dates)
test_series_pred = pd.Series(y_test_pred_original, index=test_dates)

# ============== 可视化部分 ==============
# 全部数据集的真实值
full_data = pd.concat([train_data_final, val_data, test_data], axis=0)
full_close = full_data['Close']

plt.figure(figsize=(14, 7))
plt.plot(full_close.index, full_close.values, label='True Close (All Data)', color='black')

# 标注验证与测试集起点
plt.axvline(x=val_data.index[0], color='green', linestyle='--', label='Val Start')
plt.axvline(x=test_data.index[0], color='red', linestyle='--', label='Test Start')

# 绘制验证集真实值(全范围)与预测值
plt.plot(val_data.index, val_data['Close'], label='Val True Close (All)', color='blue')
plt.plot(val_series_pred.index, val_series_pred, label='Val Predicted Close', color='blue', linestyle='--')

# 绘制测试集真实值(全范围)与预测值
plt.plot(test_data.index, test_data['Close'], label='Test True Close (All)', color='orange')
plt.plot(test_series_pred.index, test_series_pred, label='Test Predicted Close', color='orange', linestyle='--')

plt.title("Complete Data with Train/Val/Test and Predictions")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# 单独验证集图
plt.figure(figsize=(10, 5))
plt.plot(val_data.index, val_data['Close'], label='Val True Close (All)', color='blue')
plt.plot(val_series_pred.index, val_series_pred, label='Val Predicted Close', color='blue', linestyle='--')
plt.title("Validation Set Predictions (with Proper Date Index)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()

# 单独测试集图
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_data['Close'], label='Test True Close (All)', color='orange')
plt.plot(test_series_pred.index, test_series_pred, label='Test Predicted Close', color='orange', linestyle='--')
plt.title("Test Set Predictions (with Proper Date Index)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.show()
