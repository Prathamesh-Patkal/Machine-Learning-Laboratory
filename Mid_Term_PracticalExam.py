#Assignment No 5
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image, display
import graphviz

bank_data = pd.read_csv('dataset_for_5.csv', delimiter=';')

# Debug: Print column names
print("Columns in dataset:", bank_data.columns.tolist())

if 'default' in bank_data.columns:
    bank_data['default'] = bank_data['default'].map({'no': 0, 'yes': 1, 'unknown': 0})

if 'y' not in bank_data.columns:
    raise KeyError("Column 'y' not found in the dataset. Please check the dataset file.")

bank_data['y'] = bank_data['y'].map({'no': 0, 'yes': 1})

X = bank_data.drop('y', axis=1)
y = bank_data['y']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Initial Accuracy:", accuracy)

for i in range(min(3, len(rf.estimators_))):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,
                               filled=True,
                               max_depth=2,
                               impurity=False,
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(1, 20)
}

rand_search = RandomizedSearchCV(RandomForestClassifier(),
                                 param_distributions=param_dist,
                                 n_iter=5,
                                 cv=5)

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print('Best hyperparameters:', rand_search.best_params_)

y_pred = best_rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Final Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importances.plot.bar()
