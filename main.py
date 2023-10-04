import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
    st.image("logo ABB.png")
    st.title("Fraud detector")
    choice = st.radio("Navigation", ["Upload", "Profiling", "MachineTraining", "Download"])
    st.info("Cette application web permet de faire du profiling de données avec Panda et PyCaret, ainsi que l'entrainement de modèle, en ayant recours à des techniques de magie noir ainsi qu'au vodoo !")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Inserez Vos Données Pour Modélisation !")
    file = st.file_uploader("insérez votre fichier ici !")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploration Et Analyse De Donées")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

if choice == "MachineTraining":
    st.title("Entrainement Du Modèle")
    target = st.selectbox("Selectionnez Votre Cible !", df.columns)
    if st.button("Train Model"):
        setup(df, target=target, verbose = False)
        setup_df = pull()
        st.info("Ici Oppère La Magie !")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("Voici Le Modèle ML !")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')
if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Télécharger Le Modèle", f, "trained_model.pkl")

