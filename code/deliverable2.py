import pandas as pd

df = pd.read_csv("data/rentals.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("\nData Types:\n", df.dtypes)
print(df.select_dtypes(include=['object', 'bool']).nunique())

features = ['area_sqft', 'bedrooms', 'parking_spaces',
            'metro_distance_min', 'property_category',
            'property_type', 'furnishing', 'community']

X = pd.get_dummies(df[features], drop_first=True, dtype=int)
y = df['annual_rent_usd']

print("Prepared Features:", X.shape)
print("Target Variable:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train[['area_sqft']], y_train)

predictions = model.predict(X_test[['area_sqft']])

print("R2 Score:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

multi_predictions = model_multi.predict(X_test)

print("Multiple Regression R2:", r2_score(y_test, multi_predictions))
print("Multiple Regression MAE:", mean_absolute_error(y_test, multi_predictions))
print("Multiple Regression RMSE:", mean_squared_error(y_test, multi_predictions) ** 0.5)

print("\nIntercept:", model_multi.intercept_)
print("Coefficients:", pd.Series(model_multi.coef_, index=X.columns).to_dict())

df['rent_category'] = (df['annual_rent_usd'] > df['annual_rent_usd'].median()).astype(int)

y_class = df['rent_category']

print("Rent Category Distribution:\n", y_class.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

classifiers = {
    "Logistic Regression": (LogisticRegression(max_iter=1000), True),
    "KNN": (KNeighborsClassifier(), True),
    "Naive Bayes": (GaussianNB(), True),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), False)
}

for name, (clf, scaled) in classifiers.items():
    train = X_train_c_scaled if scaled else X_train_c
    test = X_test_c_scaled if scaled else X_test_c

    clf.fit(train, y_train_c)
    pred = clf.predict(test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test_c, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test_c, pred))

    best_model = classifiers["Logistic Regression"][0]
best_predictions = best_model.predict(X_test_c_scaled)

print("\nBest Model: Logistic Regression")
print("First 10 Predictions:", best_predictions[:10])
print("Actual Values:", y_test_c.values[:10])

from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

cols = ['area_sqft', 'bedrooms', 'parking_spaces',
        'metro_distance_min', 'annual_rent_usd']

data = df[cols]
scaled = StandardScaler().fit_transform(data)

df['kmeans_cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(scaled)

sample = data.sample(1000, random_state=42)
sample_scaled = StandardScaler().fit_transform(sample)
hier_clusters = AgglomerativeClustering(n_clusters=3).fit_predict(sample_scaled)

print("\nK-Means Cluster Counts:\n", df['kmeans_cluster'].value_counts().sort_index())
print("\nK-Means Cluster Profiles:\n", df.groupby('kmeans_cluster')[cols].mean().round(2))

print("\nHierarchical Cluster Counts:\n", pd.Series(hier_clusters).value_counts().sort_index())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['area_sqft'], df['annual_rent_usd'], c=df['kmeans_cluster'], s=5)
plt.title("K-Means Clustering")
plt.xlabel("Property Area (sqft)")
plt.ylabel("Annual Rent (USD)")

plt.subplot(1, 2, 2)
plt.scatter(sample['area_sqft'], sample['annual_rent_usd'], c=hier_clusters, s=8)
plt.title("Hierarchical Clustering")
plt.xlabel("Property Area (sqft)")
plt.ylabel("Annual Rent (USD)")

plt.tight_layout()
plt.show()

print("\nHierarchical Cluster Profiles:\n",
      sample.assign(cluster=hier_clusters).groupby('cluster')[cols].mean().round(2))