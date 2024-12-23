import numpy as np
import pandas as pd
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt


df = pd.read_csv("intrusion_data.csv")

X = df.drop('class', axis = 1)
y = df['class']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


base_classifier = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoost classifier with the base classifier
ada_boost = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)

# Train the Adaboost model
ada_boost.fit(X_train, y_train)

# Initialize the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the k-NN model
knn.fit(X_train, y_train)

ada_prediction = ada_boost.predict(X_test)

knn_predictions = knn.predict(X_test)

ada_accuracy = accuracy_score(y_test, ada_prediction)
knn_accuracy= accuracy_score(y_test, knn_predictions)

ada_f1 = f1_score(y_test, ada_prediction, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

print("AdaBoost Accuracy: ", ada_accuracy)
print("k-NN Accuracy: ", knn_accuracy)

print("AdaBoost F1-Score: ", ada_f1)
print("k-NN F1-Score: ", knn_f1)

Ada_Kn_model = ['AdaBoost','k-NN']
accurate = [ada_accuracy, knn_accuracy]
f1_scores = [ada_f1, knn_f1]

pl.figure(figsize=(8, 6))

plt.plot(Ada_Kn_model, accurate, marker ='o', label='Accuracy', color = 'blue', linestyle='-', linewidth= 2)

plt.plot(Ada_Kn_model, f1_scores, marker ='o', label='F1_score', color = 'red', linestyle='-', linewidth= 2)

plt.ylabel('Scores')
plt.legend()
plt.show()




