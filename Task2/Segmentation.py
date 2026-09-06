import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = pd.read_excel("customers.xlsx")

# Step 2: Select useful columns
X = data[['creditLimit','country']].copy()

# Step 3: Encode country (convert text to numbers)
X['country'] = X['country'].astype('category').cat.codes

# Step 4: Scale values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 6: Analyze clusters
print(data.groupby('Cluster')['creditLimit'].mean())

# Step 7: Visualize clusters
sns.scatterplot(x=data['creditLimit'], y=data['country'], hue=data['Cluster'], palette='Set1')
plt.show()

# Step 8: Save results
data.to_excel(r"C:\Users\Admin\OneDrive\Desktop\Thiranex Data Analytics Intership\Task2\segmented_customers.xlsx", index=False)

