import streamlit as st
import pandas as pd


def formulaire_patient(X_columns):
    st.header("🧾 Formulaire pour nouveau patient")

    champs_visibles = {
        "Epigastralgie": "Epigastralgie",
        "Metastases_Hepatiques": "Métastases hépatiques",
        "Dénutrition": "Dénutrition",
        "tabac": "Tabac",
        "Mucineux": "Mucineux",
        "Ulcéro_Bourgeonnant": "Ulcéro bourgeonnant",
        "Adenopaties": "Adénopathies",
        "Ulcere_gastrique": "Ulcère gastrique",
        "infiltrant": "Infiltrant",
        "Cardiopathie": "Cardiopathie",
        "Sténosant": "Sténosant",
    }

    user_input = {}
    for col in X_columns:
        label = champs_visibles.get(col, col)
        user_input[col] = st.number_input(f"{label}", min_value=0, max_value=1, step=1)

    return pd.DataFrame([user_input])
