import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv('insurance.csv')
data.head()

print(data.shape)
data['smoker'] = data['smoker'].apply(lambda x: 0 if x == 'no' else 1)
data['sex'] = data['sex'].apply(lambda x: 0 if x == 'female'else 1)

data = pd.get_dummies(data)
data.head()

features = data.drop('charges', axis=1).columns

X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Количество наблюдений в тестовом наборе:", X_test.shape[0])

model = linear_model.LinearRegression()
model.fit(X_train, y_train)


intercept = round(model.intercept_, 2)
print("Свободный член (intercept):", intercept)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)


r2_train = round(metrics.r2_score(y_train, y_train_pred), 3)
mae_train = round(metrics.mean_absolute_error(y_train, y_train_pred))
mape_train = round(metrics.mean_absolute_percentage_error(y_train, y_train_pred) * 100)


r2_test = round(metrics.r2_score(y_test, y_test_pred), 3)
mae_test = round(metrics.mean_absolute_error(y_test, y_test_pred))
mape_test = round(metrics.mean_absolute_percentage_error(y_test, y_test_pred) * 100)

print("Тренировочные данные - R²:", r2_train, "MAE:", mae_train, "MAPE:", mape_train)
print("Тестовые данные - R²:", r2_test, "MAE:", mae_test, "MAPE:", mape_test)


train_errors = y_train - y_train_pred
test_errors = y_test - y_test_pred

plt.figure(figsize=(10, 6))
sns.boxplot(data=[train_errors, test_errors])
plt.xticks([0, 1], ['Train', 'Test'])
plt.ylabel('Ошибка (y - ŷ)')
plt.show()


scaler = preprocessing.MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("Количество столбцов после преобразования:", X_train_poly.shape[1])

model_poly = linear_model.LinearRegression()
model_poly.fit(X_train_poly, y_train)

y_test_pred_poly = model_poly.predict(X_test_poly)
r2_test_poly = round(metrics.r2_score(y_test, y_test_pred_poly), 3)
print("R² на тестовой выборке (полиномиальная модель):", r2_test_poly)

print("Максимальный коэффициент:", round(max(model_poly.coef_), 2))
print("Минимальный коэффициент:", round(min(model_poly.coef_), 2))

lasso = linear_model.Lasso(max_iter=2000, alpha=1.0)  
lasso.fit(X_train_poly, y_train)

y_test_pred_lasso = lasso.predict(X_test_poly)

r2_lasso = round(metrics.r2_score(y_test, y_test_pred_lasso), 3)
mae_lasso = round(metrics.mean_absolute_error(y_test, y_test_pred_lasso))
mape_lasso = round(metrics.mean_absolute_percentage_error(y_test, y_test_pred_lasso) * 100)

print("Lasso - R²:", r2_lasso, "MAE:", mae_lasso, "MAPE:", mape_lasso)
