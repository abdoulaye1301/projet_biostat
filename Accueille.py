import streamlit as st
import pandas as pd
import shap as sh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import numpy as np

# from PIL import Image


st.set_page_config(page_title="NDBASY", page_icon="🎢", layout="centered")
# st.sidebar.success("Selectionnez une page")


# Definition de la fonction principale
def main():
    st.markdown(
        "**Prédiction du décès chez patient traité et suivie pour cancer de l'estomac**"
    )
    df = pd.read_excel("ccc.xlsx")
    seed = 0
    # Train/test Split
    y = df["DECES"]
    x = df.drop(
        ["DECES"],
        axis=1,
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )
    clf = MLPClassifier(
        hidden_layer_sizes=(10, 20), activation="relu", random_state=3
    ).fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    # st.write(clf.score(X_test, Y_test))
    # Créer une fonction wrapper pour le modèle
    def model_predict(X_train):
        return clf.predict_proba(X_train)[:, 1]

    def plot_perf(graphes):

        if "Metric" in graphes:
            accuracy = accuracy_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred)
            recall = recall_score(Y_test, Y_pred)
            tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
            specificity = tn / (tn + fp)

            # Affichage des métriques dans l'application
            st.write(f"Accuracy : {round(accuracy,3)}")
            # t.write(f"Precision : {round(precision,2)}")
            # st.write(f"Recall : {round(recall,2)}")
            # st.write(f"Specificity : {specificity:.2f}")

        if "Matrice de confusion" in graphes:
            st.subheader("Matrice de confusion")
            ConfusionMatrixDisplay.from_estimator(clf, X_test, Y_test)
            st.pyplot()

        if "Courbe de ROC" in graphes:
            st.markdown("**Courbe de ROC**")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(clf, X_test, Y_test, ax=ax)
            st.pyplot(fig)

        if "SHAP" in graphes:

            # Valeurs SHAP
            st.subheader("Interprétation des valeurs de SHAP")
            explainer = sh.Explainer(model_predict, X_train)
            shap_values_patient = explainer(donnee_entre)
            st.markdown(f"**K = {np.sum(shap_values_patient[0].values):.3f}**")
            shap_val = shap_values_patient[0].values
            base_val = shap_values_patient[0].base_values
            explanation = sh.Explanation(
                values=shap_val,
                base_values=base_val,
                data=donnee_entre.iloc[0],
                feature_names=donnee_entre.columns,
            )
            fig, ax = plt.subplots()
            sh.plots.waterfall(explanation, max_display=11, show=False)
            st.pyplot(fig)

    # Colletion des données d'entré
    st.sidebar.title("+ NOUVEAU PATIENT")
    st.sidebar.warning("1 = Oui, 0 = Non")

    def patient():
        colo = st.sidebar.columns(2)
        Epigastralgie = colo[0].number_input(
            "Epigastralgie", min_value=0, max_value=1, step=1
        )
        Metastases_Hepatiques = colo[1].number_input(
            "Métastases hépatiques", min_value=0, max_value=1, step=1
        )
        Dénutrition = colo[0].number_input(
            "Dénutrition", min_value=0, max_value=1, step=1
        )
        tabac = colo[1].number_input("Tabac", min_value=0, max_value=1, step=1)
        Mucineux = colo[0].number_input("Mucineux", min_value=0, max_value=1, step=1)
        Ulcéro_Bourgeonnant = colo[1].number_input(
            "Ulcéro bourgeonnant", min_value=0, max_value=1, step=1
        )
        Adenopaties = colo[0].number_input(
            "Adénopathies", min_value=0, max_value=1, step=1
        )
        Ulcere_gastrique = colo[1].number_input(
            "Ulcère gastrique", min_value=0, max_value=1, step=1
        )
        infiltrant = colo[0].number_input(
            "Infiltrant", min_value=0, max_value=1, step=1
        )
        Cardiopathie = colo[1].number_input(
            "Cardiopathie", min_value=0, max_value=1, step=1
        )
        Sténosant = st.sidebar.number_input(
            "Sténosant", min_value=0, max_value=1, step=1
        )
        donne = {
            "Epigastralgie": Epigastralgie,
            "Metastases Hepatiques": Metastases_Hepatiques,
            "Dénutrition": Dénutrition,
            "tabac": tabac,
            "Mucineux": Mucineux,
            "Ulcéro Bourgeonnant": Ulcéro_Bourgeonnant,
            "Adenopaties": Adenopaties,
            "Ulcere gastrique": Ulcere_gastrique,
            "infiltrant": infiltrant,
            "Cardiopathie": Cardiopathie,
            "Sténosant": Sténosant,
        }
        donneePatient = pd.DataFrame(donne, index=[0])
        return donneePatient

    donne2 = patient()

    # Tranformation des données d'entré

    donnee_ent = pd.concat([donne2, x], axis=0)
    # Récupération de la première ligne
    donnee_entre = donnee_ent[:1]

    # Affichage des données transformé
    if st.sidebar.button("Prediction"):
        # Prévision
        prevision = clf.predict(donnee_entre)
        # Affichage du prévision
        st.subheader("Résultat du modèle de MLPClassifier")
        # Affichage des graphiques de performance
        plot_perf(["Metric", "Courbe de ROC", "SHAP"])


if __name__ == "__main__":
    main()
