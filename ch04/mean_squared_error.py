#3.6の例を引き継いで
import numpy as np
#正解データ(one-hot表現)
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#"2"の確率が0.6
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

print(mean_squared_error(np.array(y), np.array(t)))

#"7"の確率が0.6
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y), np.array(t)))
