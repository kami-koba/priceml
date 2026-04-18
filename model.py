import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# === ÉTAPE 1 : Charger les données ===
df = pd.read_csv('CarPrice_Assignment.csv')

# === ÉTAPE 2 : Sélectionner les variables ===
features = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 
            'enginesize', 'horsepower', 'citympg', 'highwaympg']

X = df[features]
y = df['price']

print(f"X : {X.shape}")
print(f"y : {y.shape}")

# === ÉTAPE 3 : Séparer train / test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nEntraînement : {X_train.shape[0]} lignes")
print(f"Test : {X_test.shape[0]} lignes")

# === ÉTAPE 4 : Entraîner le modèle ===
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModèle entraîné !")

# === ÉTAPE 5 : Évaluer ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== RÉSULTATS ===")
print(f"R²   : {r2:.4f}")
print(f"RMSE : {rmse:.2f} $")

# === ÉTAPE 6 : Coefficients ===
print(f"\n=== COEFFICIENTS ===")
print(f"Intercept : {model.intercept_:.2f}")
for col, coef in zip(features, model.coef_):
    print(f"{col:20s} : {coef:.4f}")