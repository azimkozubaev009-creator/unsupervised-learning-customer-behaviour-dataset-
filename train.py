import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'Customer_Data.csv')
model_path = os.path.join(BASE_DIR, 'model', 'kmeans_model.pkl')

df = pd.read_csv(data_path)
df = df.dropna()

X = df[['Annual_Income', 'Spending_Score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=5, random_state=42)
model.fit(X_scaled)
    
centers = model.cluster_centers_

if not os.path.exists(os.path.join(BASE_DIR, 'model')):
    os.makedirs(os.path.join(BASE_DIR, 'model'))

joblib.dump({
    "model": model,
    "scaler": scaler,
    "centers": centers
}, model_path)

print("Model trained correctly with scaling!")
