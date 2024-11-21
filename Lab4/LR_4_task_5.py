import pickle
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Створення даних
m = 100
X = np.linspace(-3, 3, m)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, m)

X = X.reshape(-1, 1)

# Розділення даних
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Лінійна регресія
linear_model_reg = linear_model.LinearRegression()
linear_model_reg.fit(X_train, y_train)

# Прогноз лінійної регресії
y_linear_pred = linear_model_reg.predict(X)

# Побудова графіка для лінійної регресії
plt.scatter(X, y, color='blue', s=10, label='Дані')  # Крапки
plt.plot(X, y_linear_pred, color='green', linewidth=2, label='Лінійна регресія')  # Лінійна крива
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Лінійна регресія")
plt.show()

# Поліноміальна регресія
degree = 2
poly_model = make_pipeline(PolynomialFeatures(degree), linear_model.LinearRegression())
poly_model.fit(X_train, y_train)

# Прогноз поліноміальної регресії
y_poly_pred = poly_model.predict(X)

# Побудова графіка для поліноміальної регресії
plt.scatter(X, y, color='blue', s=10, label='Дані')  # Крапки
plt.plot(X, y_poly_pred, color='red', linewidth=2, label='Поліноміальна регресія')  # Поліноміальна крива
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Поліноміальна регресія")
plt.show()