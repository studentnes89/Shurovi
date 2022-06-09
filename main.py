import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv("data_one.csv")

##Выбираем регион
df_new = df[["region", "name", "city", "price", "cuisine"]]
Region = st.selectbox(
        "Region", df_new["region"].value_counts().index
    )
df_selection = df_new[lambda x: x["region"] == Region]
df_selection

##Выбираем кухню
Cuisine = st.selectbox(
        "Cuisine", df_new["cuisine"].value_counts().index
    )
df_selection2 = df_new[lambda x: x["cuisine"] == Cuisine]
df_selection2
