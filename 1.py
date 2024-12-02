import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 生成随机时间序列数据
#将time_series换成csv文件中的一列
data = pd.read_csv(r'E:\MS\data\eadro-main\eadro-main\data\SN_process\data\observe\data_pre\abnormal.csv')
time_series = data['compose-post-service_tx_bytes'].values
#time_series标准化
time_series = (time_series - time_series.mean()) / time_series.std()
# 计算滑动窗口标准差
window_size = 3
rolling_std = np.array([np.std(time_series[max(0, i-window_size):i+1]) for i in range(len(time_series))])

#计算一阶差分
diff = np.diff(time_series)
# 绘制时间序列、一阶差分和滑动窗口标准差
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_series, label='Time Series')
plt.title('Random Time Series with Sudden Changes')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(diff, label='Time Series')
plt.title('Random Time Series with Sudden Changes')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(rolling_std, label='Rolling Std Dev', color='purple')
plt.title('Rolling Standard Deviation of Time Series')
plt.legend()

plt.tight_layout()
plt.show()