import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv("data_one.csv")


df_new = df[["region", "name", "city", "price", "cuisine"]]
Region = st.selectbox(
        "Region", df_new["region"].value_counts().index
    )
df_selection = df_new[lambda x: x["region"] == Region]
df_selection
