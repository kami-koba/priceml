import pandas as pd

df = pd.read_csv('CarPrice_Assignment.csv')

print("=== DIMENSIONS ===")
print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes")

print("\n=== COLONNES ===")
print(df.columns.tolist())

print("\n=== APERÇU ===")
print(df.head())

print("\n=== TYPES DE DONNÉES ===")
print(df.dtypes)

print("\n=== VALEURS MANQUANTES ===")
print(df.isnull().sum())