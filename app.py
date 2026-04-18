import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import anthropic

# === Configuration de la page ===
st.set_page_config(
    page_title="PriceML",
    page_icon="🚗",
    layout="centered"
)

# === Titre ===
st.title("🚗 PriceML")
st.markdown("**Estimation de prix de voiture par régression linéaire multiple**")
st.divider()

# === Chargement et entraînement du modèle ===
@st.cache_resource
def load_model():
    df = pd.read_csv('CarPrice_Assignment.csv')
    features = ['wheelbase', 'carlength', 'carwidth', 'curbweight',
                'enginesize', 'horsepower', 'citympg', 'highwaympg']
    X = df[features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, features

model, features = load_model()

# === Sliders utilisateur ===
st.subheader("Caractéristiques de la voiture")

col1, col2 = st.columns(2)

with col1:
    wheelbase = st.slider("Empattement (cm)", 85, 120, 100)
    carlength = st.slider("Longueur (cm)", 140, 210, 170)
    carwidth = st.slider("Largeur (cm)", 60, 75, 67)
    curbweight = st.slider("Poids (kg)", 1500, 4000, 2500)

with col2:
    enginesize = st.slider("Cylindrée (cm³)", 60, 330, 150)
    horsepower = st.slider("Puissance (cv)", 48, 288, 100)
    citympg = st.slider("Conso ville (mpg)", 13, 49, 25)
    highwaympg = st.slider("Conso route (mpg)", 16, 54, 30)

# === Prédiction ===
st.divider()

if st.button("🔍 Estimer le prix", use_container_width=True):

    input_data = np.array([[wheelbase, carlength, carwidth, curbweight,
                            enginesize, horsepower, citympg, highwaympg]])
    prix_predit = model.predict(input_data)[0]

    st.success(f"💰 Prix estimé : **${prix_predit:,.0f}**")

    # === Analyse Claude ===
    st.subheader("📊 Analyse IA")
    with st.spinner("Claude analyse les résultats..."):
        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        prompt = f"""
        Un modèle de régression linéaire multiple a prédit le prix d'une voiture.
        
        Caractéristiques entrées :
        - Empattement : {wheelbase} cm
        - Longueur : {carlength} cm
        - Largeur : {carwidth} cm
        - Poids : {curbweight} kg
        - Cylindrée : {enginesize} cm³
        - Puissance : {horsepower} cv
        - Consommation ville : {citympg} mpg
        - Consommation route : {highwaympg} mpg
        
        Prix prédit : ${prix_predit:,.0f}
        
        En 3 phrases maximum, explique ce résultat de façon simple et professionnelle.
        Mentionne les caractéristiques qui influencent le plus le prix.
        Réponds en français.
        """
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        st.write(message.content[0].text)