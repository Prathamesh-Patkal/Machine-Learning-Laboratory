from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, marker="*")
plt.title("Synthetic Data (2 Features)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

model = GaussianNB()
model.fit(X_train, y_train)

predicted = model.predict([X_test[6]])
print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print("Accuracy:", accuracy)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Synthetic Data")
plt.show()

csv_file = 'dataset_for_6.csv'

if not os.path.exists(csv_file):
    print(f"File not found: {csv_file}")
else:
    df = pd.read_csv(csv_file)
    print("\n Dataset Loaded Successfully.")
    print("First 5 rows:")
    print(df.head())

    print("\nColumns in dataset:", df.columns.tolist())

    if 'purpose' in df.columns and 'not.fully.paid' in df.columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x='purpose', hue='not.fully.paid')
        plt.xticks(rotation=45, ha='right')
        plt.title("Loan Purpose vs Not Fully Paid")
        plt.show()
    else:
        print("Required columns 'purpose' and/or 'not.fully.paid' not found.")

    if 'purpose' in df.columns:
        pre_df = pd.get_dummies(df, columns=['purpose'], drop_first=True)
    else:
        pre_df = df.copy()

    if 'not.fully.paid' in pre_df.columns:
        X = pre_df.drop('not.fully.paid', axis=1)
        y = pre_df['not.fully.paid']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=125
        )

        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("\n--- Used Cars Dataset ---")
        print("Accuracy:", accuracy)
        print("F1 Score:", f1)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix - Used Cars Data")
        plt.show()
    else:
        print("Target column 'not.fully.paid' not found.")
