import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

FILE1_PATH = "./models/resnet18_training_history.csv"
FILE2_PATH = "./models/mobilenet_v3_small_training_history.csv"

PATHS = [FILE1_PATH, FILE2_PATH]

def get_training_dataframe(path_str):
    training_data_1 = pd.read_csv(path_str)
    st.dataframe(training_data_1)

    return training_data_1["architecture"][0], training_data_1

def select_model(model_names):
    option= st.selectbox( "Select the model", options=model_names)
    return option

def plot_matplotlib(selected_df):
    """Optional"""
    fig, ax = plt.subplots(figsize=(5,6))
    metric = "val_loss"
    y_val = list(selected_df[metric])
    ax.plot(selected_df["epoch"],y_val )
    min_val = np.min(y_val)
    max_val = np.max(y_val)
    ax.set_ylim(min_val, max_val)
    st.pyplot(fig)


def main():
    st.title("ML Dashboard")
    st.text("This is my first app.")
   
    a = np.linspace(0,10, 6)
    st.text(a)

    # Collecting model_name: model_df in a dictionary
    models_storage = {}
    for path_str in PATHS:
        model_name, model_df = get_training_dataframe(path_str)
        models_storage[model_name] = model_df
    
    # Extract all model_names into a list so I can get the selection
    model_names = list(models_storage.keys())
    selected_option = select_model(model_names)

    st.subheader(f" {selected_option}: Loss during training history")
    
    # Pass the selected object to the dictionary to access the dataframe
    selected_df = models_storage[selected_option]
    # How do I access the right training history from the name alone?
    st.line_chart(selected_df, x = "epoch", y=["val_loss", "train_loss"])

    df_columns = list(selected_df.columns)

    metrics = [column for column in df_columns if (("val" in column) or ("train" in column)) ]
    selected_metric = st.selectbox("Select a metric to compare across all models", options=metrics)


    # Implement 2nd widget where you select the metric and show all
    # the behaviour of that metric during training (x="epochs") for all available models 
    # train loss, train accuracy, val loss, val accuracy


    


if __name__ == "__main__":
    main()
