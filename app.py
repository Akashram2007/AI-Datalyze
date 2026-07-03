import streamlit as st
import pandas as pd

from modules.preview import data_preview
from modules.summary import data_summary
from modules.cleaning import data_cleaning
from modules.visualization import visualization
from modules.prediction import prediction

st.set_page_config(page_title="AI Datalyze", page_icon="🔍")
st.sidebar.title("AI Datalyze")
st.sidebar.divider()
file = st.file_uploader("Upload File", type=["csv", "xlsx"], width=400)
menu = st.sidebar.radio(
    "Sections :",
    options=[
        "🔍Data Preview",
        "📈Data Summary",
        "🧹Missing Data/Cleaning",
        "📊Visualization",
        "⚙️Prediction",
    ],
)

if file is not None:
    file_type = file.name.split(".")[-1]
    if file_type == "csv":
        data = pd.read_csv(file)
    elif file_type == "xlsx":
        data = pd.read_excel(file, engine="openpyxl")
    else:
        st.error("Unsupported File Type")

    data.index = range(1, len(data) + 1)
    data.index.name = "S.No"

    st.divider()

    st.title(menu)
    # ============================DATA PREVIEW============================

    if menu == "🔍Data Preview":
        data_preview(data, file)

    # ============================DATA SUMMARY============================

    if menu == "📈Data Summary":
        data_summary(data)

    # ============================MISSING/CLEANING DATA===================

    if menu == "🧹Missing Data/Cleaning":
        data_cleaning(data)

    # ============================VISUALIZATION============================
    if menu == "📊Visualization":
        visualization(data)
    
    #============================Prediction================================
    if menu == "⚙️Prediction":
        prediction(data)
        
else:
    st.title("AI Datalyze")
    st.divider()
    st.header("Analyze, Clean, and Visualize Your Data in Seconds")
    st.subheader("Start Exploring Your Data Now!!")
    st.title("🔍📈🧹📊")
    st.error("Upload your File")
