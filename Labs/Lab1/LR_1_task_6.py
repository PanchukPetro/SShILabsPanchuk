import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Завантажуємо дані з файлу
data = pd.read_csv('data_multivar_nb.txt', header=None)
X = data.iloc[:, :-1].values  # ознаки (фічі)
y = data.iloc[:, -1].values   # цільові мітки

# Розділяємо дані на тренувальні та тестові набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Класифікація за допомогою машини опорних векторів (SVM)
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Оцінка якості моделі SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

# 2. Класифікація за допомогою наївного байєсівського класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Оцінка якості моделі Наївного Байєса
accuracy_nb = accuracy_score(y_test, y_pred_nb)
recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

# Виведення результатів
print("SVM Classifier Results:")
print(f"Accuracy: {accuracy_svm:.3f}")
print(f"Recall: {recall_svm:.3f}")
print(f"Precision: {precision_svm:.3f}")
print(f"F1 Score: {f1_svm:.3f}")
print("")

print("Naive Bayes Classifier Results:")
print(f"Accuracy: {accuracy_nb:.3f}")
print(f"Recall: {recall_nb:.3f}")
print(f"Precision: {precision_nb:.3f}")
print(f"F1 Score: {f1_nb:.3f}")

# Порівняння результатів та висновки
if f1_svm > f1_nb:
    print("\nМодель SVM краще підходить для цієї задачі класифікації.")
elif f1_svm == f1_nb:
    print("\nМоделі показують однакову продуктивність")
else:
    print("\nМодель наївного байєсівського класифікатора краще підходить для цієї задачі класифікації.")