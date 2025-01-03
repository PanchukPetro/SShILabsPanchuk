import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Використовуємо для нашого аналізу дані, що містяться у файлі data_random_forests.txt.
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розіб'ємо дані на три класи.
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

# Розіб'ємо дані на навчальний та тестовий набори.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Задамо сітку значень параметрів, де будемо тестувати класифікатор.
parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]},
]

# Визначимо метрики, які використовуватимемо для оцінки.
metrics = ['precision_weighted', 'recall_weighted']

# Виконуємо пошук оптимальних параметрів для кожної метрики.
for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid,
        cv=5,
        scoring=metric
    )
    classifier.fit(X_train, y_train)

print("\nGrid scores for the parameter grid")
results = classifier.cv_results_
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(params, '-->', mean_score)

print("\nBest parameters:", classifier.best_params_)

y_pred = classifier.predict(X_test)
print("\n Performance report: \n")
print(classification_report(y_test,y_pred))

