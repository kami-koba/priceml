import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# === Charger et préparer ===
df = pd.read_csv('CarPrice_Assignment.csv')
features = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 
            'enginesize', 'horsepower', 'citympg', 'highwaympg']
X = df[features]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# === GRAPHE 1 : Réel vs Prédit ===
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='steelblue')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Prédiction parfaite')
plt.xlabel('Prix réel ($)')
plt.ylabel('Prix prédit ($)')
plt.title('Prix réel vs Prix prédit')
plt.legend()
plt.tight_layout()
plt.savefig('graphe_reel_vs_predit.png')
plt.show()
print("Graphe 1 sauvegardé")

# === GRAPHE 2 : Coefficients ===
coefs = pd.Series(model.coef_, index=features).sort_values()
colors = ['red' if c < 0 else 'steelblue' for c in coefs]
plt.figure(figsize=(7, 5))
coefs.plot(kind='barh', color=colors)
plt.axvline(x=0, color='black', linewidth=0.8)
plt.title('Impact de chaque variable sur le prix')
plt.xlabel('Coefficient')
plt.tight_layout()
plt.savefig('graphe_coefficients.png')
plt.show()
print("Graphe 2 sauvegardé")

# === GRAPHE 3 : Résidus ===
residus = y_test - y_pred
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residus, alpha=0.7, color='steelblue')
plt.axhline(y=0, color='red', linewidth=2, linestyle='--')
plt.xlabel('Prix prédit ($)')
plt.ylabel('Résidu ($)')
plt.title('Analyse des résidus')
plt.tight_layout()
plt.savefig('graphe_residus.png')
plt.show()
print("Graphe 3 sauvegardé")