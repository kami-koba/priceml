# 🚗 PriceML

Application de Machine Learning appliquée à l'estimation de prix et à la 
segmentation de véhicules, assistée par intelligence artificielle (Claude API).

👉 **[Demo live](https://priceml-q8bmdoldejfvjvteqqiktf.streamlit.app)**

## Contexte

Projet développé pour démontrer l'application du Machine Learning à la 
prédiction de prix et à l'analyse exploratoire de données techniques — 
cas d'usage directement transposable à la tarification d'équipements 
industriels sur mesure.

## Fonctionnalités

### Onglet 1 — Prédiction de prix
- Saisie des caractéristiques d'une voiture via sliders interactifs
- Prédiction du prix par régression linéaire multiple (scikit-learn)
- Analyse en langage naturel des résultats générée par Claude AI
- Modèle entraîné sur 205 véhicules réels (R² = 0.82, RMSE = 3 802$)

### Onglet 2 — Clustering
- Segmentation automatique des véhicules en 3 groupes (K-Means)
- Visualisation des segments : entrée / milieu / haut de gamme
- Tableau récapitulatif des caractéristiques moyennes par segment

## Stack technique

- Python · scikit-learn · Pandas · NumPy
- Streamlit · Anthropic Claude API · Matplotlib

## Dataset

[Car Price Prediction — Kaggle](https://www.kaggle.com/datasets/hellbuoy/car-price-prediction)

## Lancer en local

```bash
git clone https://github.com/kami-koba/priceml.git
cd priceml
pip install -r requirements.txt
streamlit run app.py
```

## Résultats du modèle

| Métrique | Valeur |
|---|---|
| R² | 0.82 |
| RMSE | 3 802 $ |
| Observations | 205 |
| Segments K-Means | 3 |

## Auteur

**Koba Kami Victor** — [LinkedIn](https://linkedin.com/in/kami-koba) · 
[GitHub](https://github.com/kami-koba)
