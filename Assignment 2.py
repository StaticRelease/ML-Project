import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

#Using pd.read read the file and removing the labels to set them as X and Y
data = pd.read_csv("file_pe.csv")
X=data.drop(["Malware", "Name"], axis=1)
y=data["Malware"]

# calculates the mean and standard deviation
X_standardized=StandardScaler().fit_transform(X)
#Initializes Logistic Regression
lr_model = LogisticRegression(solver='liblinear', random_state=123)
rfe_model = RFE(estimator=lr_model, n_features_to_select=10, step=1)
rfe_model.fit(X_standardized, y)


selected_features = rfe_model.support_
X_selected = X_standardized[:, selected_features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=23)

model = LogisticRegression(solver='liblinear', random_state=123)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)