import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import anthropic

st.set_page_config(page_title="PriceML", page_icon="🚗", layout="centered")

# === Chargement du modèle ===
@st.cache_resource
def load_model():
    df = pd.read_csv('CarPrice_Assignment.csv')
    features = ['wheelbase', 'carlength', 'carwidth', 'curbweight',
                'enginesize', 'horsepower', 'citympg', 'highwaympg']
    X = df[features]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, features, df

model, features, df = load_model()

# === Navigation ===
st.title("🚗 PriceML")
st.markdown("**Estimation de prix et segmentation de véhicules par ML**")
st.divider()

onglet1, onglet2 = st.tabs(["📈 Prédiction de prix", "🔵 Clustering"])

# ==============================
# ONGLET 1 — RÉGRESSION
# ==============================
with onglet1:
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

    st.divider()
    if st.button("🔍 Estimer le prix", use_container_width=True):
        input_data = np.array([[wheelbase, carlength, carwidth, curbweight,
                                enginesize, horsepower, citympg, highwaympg]])
        prix_predit = model.predict(input_data)[0]
        st.success(f"💰 Prix estimé : **${prix_predit:,.0f}**")

        st.subheader("📊 Analyse IA")
        with st.spinner("Claude analyse les résultats..."):
            client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
            prompt = f"""
            Un modèle de régression linéaire multiple a prédit le prix d'une voiture.
            Caractéristiques : empattement {wheelbase}cm, longueur {carlength}cm,
            largeur {carwidth}cm, poids {curbweight}kg, cylindrée {enginesize}cm³,
            puissance {horsepower}cv, conso ville {citympg}mpg, conso route {highwaympg}mpg.
            Prix prédit : ${prix_predit:,.0f}
            En 3 phrases max, explique ce résultat simplement en français.
            Mentionne les caractéristiques qui influencent le plus le prix.
            """
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            st.write(message.content[0].text)

# ==============================
# ONGLET 2 — CLUSTERING
# ==============================
with onglet2:
    st.subheader("Segmentation automatique des véhicules")
    st.markdown("K-Means détecte automatiquement 3 segments dans les données.")

    features_clust = ['enginesize', 'horsepower', 'curbweight', 'price']
    data = df[features_clust].dropna()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['cluster'] = kmeans.fit_predict(data_scaled)

    # Résumé
    resume = data.groupby('cluster')[features_clust].mean().round(0)
    noms = {0: "Entrée de gamme", 1: "Milieu de gamme", 2: "Haut de gamme"}
    resume.index = [noms[i] for i in resume.index]
    st.dataframe(resume, use_container_width=True)

    # Graphe
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e94560', '#53d8fb', '#4caf50']
    for i in range(3):
        subset = data[data['cluster'] == i]
        ax.scatter(subset['horsepower'], subset['price'],
                   c=colors[i], label=noms[i], alpha=0.7)
    ax.set_xlabel('Puissance (cv)')
    ax.set_ylabel('Prix ($)')
    ax.set_title('Segmentation K-Means — 3 segments de véhicules')
    ax.legend()
    st.pyplot(fig)