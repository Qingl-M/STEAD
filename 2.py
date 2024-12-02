import numpy as np
import matplotlib.pyplot as plt

# 生成随机时间序列数据
np.random.seed(0)
time_series = np.random.normal(loc=0, scale=1, size=100)

# 添加骤升、骤降的数据点
time_series[20] += 10
time_series[50] -= 10
time_series[80] += 15

# 计算滑动窗口标准差
window_size = 5
rolling_std = np.array([np.std(time_series[max(0, i-window_size):i+1]) for i in range(len(time_series))])

# 绘制时间序列和滑动窗口标准差
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_series, label='Time Series')
plt.title('Random Time Series with Sudden Changes')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(rolling_std, label='Rolling Std Dev', color='purple')
plt.title('Rolling Standard Deviation of Time Series')
plt.legend()

plt.tight_layout()
plt.show()