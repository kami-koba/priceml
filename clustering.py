import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === Charger les données ===
df = pd.read_csv('CarPrice_Assignment.csv')

features = ['enginesize', 'horsepower', 'curbweight', 'price']
data = df[features].dropna()

# === Normaliser les données ===
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# === K-Means : 3 groupes ===
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
data['cluster'] = kmeans.fit_predict(data_scaled)

# === Résumé des groupes ===
print("=== RÉSUMÉ DES GROUPES ===")
print(data.groupby('cluster')[features].mean().round(0))

# === Visualisation ===
colors = ['#e94560', '#53d8fb', '#4caf50']
labels = ['Groupe 0', 'Groupe 1', 'Groupe 2']

plt.figure(figsize=(8, 5))
for i in range(3):
    subset = data[data['cluster'] == i]
    plt.scatter(subset['horsepower'], subset['price'],
                c=colors[i], label=labels[i], alpha=0.7)

plt.xlabel('Puissance (cv)')
plt.ylabel('Prix ($)')
plt.title('Clustering K-Means — 3 segments de véhicules')
plt.legend()
plt.tight_layout()
plt.savefig('graphe_clustering.png')
plt.show()
print("Graphe sauvegardé !")