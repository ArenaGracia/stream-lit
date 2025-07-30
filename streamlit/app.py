import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static

# Chargement modèle
model = joblib.load("model.joblib")
le = joblib.load("le.joblib")

# Coordonnées quartiers
df_coords = pd.read_csv("quartier.csv")
print(df_coords)

st.title("💰 Estimation du loyer mensuel")

# --- Entrées utilisateur ---
quartier = st.selectbox("Quartier", df_coords["quartier"].unique())
superficie = st.number_input("Superficie (m²)", min_value=10, max_value=1000, step=5)
chambres = st.slider("Nombre de chambres", 1, 10, 2)
douche_wc = st.selectbox("Salle de bain", ["intérieur", "extérieur"])
acces = st.selectbox("Type d'accès", ["sans", "moto", "voiture", "voiture_avec_par_parking"])
meuble = st.radio("Meublé", ["oui", "non"])
etat = st.selectbox("État général", ["bon", "moyen", "mauvais"])

def encoder_input():
    data = {
        "quartier": le.fit_transform([quartier])[0],
        "superficie": superficie,
        "nombre_chambres": chambres,
        "douche_wc": le.fit_transform([douche_wc])[0],
        "type_d_acces": le.fit_transform([acces])[0],
        "meublé": le.fit_transform([meuble])[0],
        "etat_general": le.fit_transform([etat])[0]
    }
    return pd.DataFrame([data])



input_df = encoder_input()
# input_scaled = model.transform(input_df)

# --- Prédiction ---
if st.button("Prédire le loyer"):
    prediction = model.predict(input_df)[0]
    st.success(f"Loyer estimé : **{round(prediction, 2)} €**")

    # --- Carte interactive ---
    row = df_coords[df_coords["quartier"] == quartier].iloc[0]
    map = folium.Map(location=[row["lat"], row["lon"]], zoom_start=15)
    folium.Marker([row["lat"], row["lon"]],
                  tooltip=f"{quartier}",
                  popup=f"Loyer estimé : {round(prediction, 2)} €").add_to(map)
    st.subheader("Localisation estimée")
    folium_static(map)

    # --- Coefficients du modèle ---
    st.subheader("📊 Importance des variables")
    coef = model.coef_
    cols = input_df.columns
    df_coef = pd.DataFrame({"Variable": cols, "Poids": coef})
    st.bar_chart(df_coef.set_index("Variable"))