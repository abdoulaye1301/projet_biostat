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


st.set_page_config(page_title="NDAO_BA", page_icon="🎢", layout="centered")
# st.sidebar.success("Selectionnez une page")


# Definition de la fonction principale
def main():
    st.markdown("")
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

    # Créer un objet explicatif explicateur
    explainer = sh.Explainer(model_predict, X_train)

    def plot_perf(graphes):

        if "Metric" in graphes:
            accuracy = accuracy_score(Y_test, Y_pred)
            precision = precision_score(Y_test, Y_pred)
            recall = recall_score(Y_test, Y_pred)
            tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
            specificity = tn / (tn + fp)

            # Affichage des métriques dans l'application
            st.write(f"Accuracy : {round(accuracy,2)}")
            st.write(f"Precision : {round(precision,2)}")
            st.write(f"Recall : {round(recall,2)}")
            st.write(f"Specificity : {specificity:.2f}")

        if "Matrice de confusion" in graphes:
            st.subheader("Matrice de confusion")
            ConfusionMatrixDisplay.from_estimator(clf, X_test, Y_test)
            st.pyplot()

        if "Courbe de ROC" in graphes:
            st.subheader("Courbe de ROC")
            # fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(clf, X_test, Y_test)
            st.pyplot()

        if "SHAP" in graphes:
            # Valeurs SHAP
            st.subheader("Les valeurs de SHAP")
            sample_idx = 55
            shap_values_55 = explainer(X_test.iloc[sample_idx : sample_idx + 1, :])

            shap_values_array = shap_values_55.values

            shap_df = np.sum(np.abs(shap_values_array), axis=0)

            shap_df = pd.DataFrame({"Feature": X_test.columns, "SHAP Score": shap_df})
            st.write(shap_df)

    # Performance du modele
    perf_graphe = st.sidebar.multiselect(
        "Choisir un graphique de performance du modèle",
        ("Metric", "Matrice de confusion", "Courbe de ROC", "SHAP"),
    )
    if st.sidebar.button("Execution", key="classify"):
        st.subheader("Résultat du modèle de MLPClassifier")
        # Affichage des graphiques de performance
        plot_perf(perf_graphe)


if __name__ == "__main__":
    main()
