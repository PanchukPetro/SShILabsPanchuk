import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(url)

#Choosing necessary fields in table
data = data[['price', 'train_type', 'fare', 'origin', 'destination']]
data = data.dropna()

#Turning price into binary value
threshold = data['price'].median()
data['price_category'] = (data['price'] > threshold).astype(int)
categorical_columns = ['train_type', 'fare', 'origin', 'destination']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

X = data.drop(columns=['price', 'price_category'])
y = data['price_category']

#TrainTestSplit and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

#Confusion matrix visualisation
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cheap', 'Expensive'], yticklabels=['Cheap', 'Expensive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()