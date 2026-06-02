import streamlit as st
import pandas as pd

FILE1_PATH = ".\models\\resnet18_training_history.csv"
FILE2_PATH = ".\models\\mobilenet_v3_small_training_history.csv"

PATHS = [FILE1_PATH, FILE2_PATH]

def get_training_dataframe(path_str):
    training_data_1 = pd.read_csv(path_str)
    st.dataframe(training_data_1)




