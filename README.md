# 🚗 PriceML

Application d'estimation de prix de voiture par régression linéaire multiple, 
assistée par intelligence artificielle (Claude API).

👉 **[Demo live](https://kami-koba-priceml.streamlit.app)**

## Contexte

Projet développé pour démontrer l'application du Machine Learning 
à la prédiction de prix d'objets techniques à partir de leurs 
caractéristiques physiques — cas d'usage directement transposable 
à la tarification d'équipements industriels sur mesure.

## Fonctionnalités

- Saisie des caractéristiques d'une voiture via sliders interactifs
- Prédiction du prix par régression linéaire multiple (scikit-learn)
- Analyse en langage naturel des résultats générée par Claude AI
- Modèle entraîné sur 205 véhicules réels (R² = 0.82, RMSE = 3802$)

## Stack technique

- Python · scikit-learn · Pandas · NumPy
- Streamlit · Anthropic Claude API

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
| Variables | 8 |

## Auteur

**Koba Kami Victor** — [LinkedIn](https://linkedin.com/in/kami-koba) · 
[GitHub](https://github.com/kami-koba)
