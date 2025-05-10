import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------- PART 1: Custom K-Means on clustering.csv ----------
# Load dataset with appropriate column names
col_names = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
             'Credit_History', 'Property_Area', 'Loan_Status']
data = pd.read_csv('clustering.csv', header=None, names=col_names, skiprows=1)

# Select features and drop missing values
X = data[["LoanAmount", "ApplicantIncome"]].dropna()

# Visualise data points
plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c='black')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.title('Original Data Points')
plt.show()

# Initialize centroids
K = 3
Centroids = X.sample(n=K)

# Plot initial centroids
plt.scatter(X["ApplicantIncome"], X["LoanAmount"], c='black')
plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"], c='red')
plt.xlabel('AnnualIncome')
plt.ylabel('Loan Amount (In Thousands)')
plt.title('Initial Centroids')
plt.show()

# Run custom K-means algorithm
diff = 1
j = 0
while diff != 0:
    XD = X.copy()
    i = 1
    for index1, row_c in Centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["ApplicantIncome"] - row_d["ApplicantIncome"])**2
            d2 = (row_c["LoanAmount"] - row_d["LoanAmount"])**2
            d = np.sqrt(d1 + d2)
            ED.append(d)
        X[i] = ED
        i = i + 1

    C = []
    for index, row in X.iterrows():
        distances = row[range(1, K+1)]
        cluster = distances.idxmin()
        C.append(cluster)
    X["Cluster"] = C

    Centroids_new = X.groupby(["Cluster"]).mean()[["LoanAmount", "ApplicantIncome"]]
    if j == 0:
        diff = 1
        j = j + 1
    else:
        diff = (Centroids_new - Centroids).sum().sum()
        print(f"Iteration {j} - Difference: {diff}")
        j = j + 1
    Centroids = Centroids_new

# Final plot
colors = ['blue', 'green', 'cyan']
for k in range(K):
    data_k = X[X["Cluster"] == k+1]
    plt.scatter(data_k["ApplicantIncome"], data_k["LoanAmount"], c=colors[k])
plt.scatter(Centroids["ApplicantIncome"], Centroids["LoanAmount"], c='red', marker='x')
plt.xlabel('Income')
plt.ylabel('Loan Amount (In Thousands)')
plt.title('Final Clusters')
plt.show()

# ---------- PART 2: Sklearn KMeans on diabetes_for_Assignment3.csv ----------
# Load diabetes dataset
data = pd.read_csv("diabetes_for_Assignment3.csv")

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pd.DataFrame(data_scaled).describe()

# Initial KMeans
kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10)
kmeans.fit(data_scaled)

# Inertia for initial model
print("Initial inertia with 2 clusters:", kmeans.inertia_)

# Elbow method to determine optimal clusters
SSE = []
for cluster in range(1, 20):
    kmeans = KMeans(n_clusters=cluster, init='k-means++', n_init=10)
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': SSE})
plt.figure(figsize=(12, 6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Final model with 5 clusters
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10)
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
print(frame['cluster'].value_counts())
