from random import random

import pandas as pd
from scipy.cluster.hierarchy import average
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pandas import json_normalize
from sklearn.preprocessing import LabelEncoder

# Take 2 json cloudtrail data sets and normalizing them to turn the strings into numbers
data1 = pd.read_json('flaws_cloudtrail14.json')
data2 = pd.read_json('flaws_cloudtrail15.json')
data1_normalized = json_normalize(data1['Records'])
data2_normalized = json_normalize(data2['Records'])

# combining both into one combined data set
combined_data = pd.concat([data1_normalized, data2_normalized], ignore_index=True)

# set the x and y values using the user agent eventsource and source IP with the event name
X = combined_data[['userAgent', 'eventSource', 'sourceIPAddress']]
y = combined_data['eventName']

# convert categorical values into numeric
label_encoder = LabelEncoder()
# apply label encoder to both x and y
X_encoded = X.apply(label_encoder.fit_transform)
y_encoded = label_encoder.fit_transform(y)

# split the encoded data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.5, random_state=42)

# set the 5 models
five_models = [
    ('logreg', LogisticRegression(max_iter=1000, random_state=42)),
    ('svm', SVC(probability=True, kernel='rbf', random_state=42)),
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('neural_net', MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)),
]

# Store results for each model
result = []
# iterate the list of models to train them
for model_name, model in five_models:
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

# calculate accuracy, precision, recall and f1
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    result.append({'model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1 })

# Create a DataFrame to store and display the results
result_DF = pd.DataFrame(result)

print(result_DF)