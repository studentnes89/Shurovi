import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv("data_one.csv")
df2=df.dropna()
df2['price'] = df2['price'].str.replace('$','a')


##Выбираем регион
df_new = df2[["region", "name", "city", "price", "cuisine"]]
Region = st.selectbox(
        "Region", df_new["region"].value_counts().index
    )
df_selection = df_new[lambda x: x["region"] == Region]
df_selection_sort = df_selection.sort_values('price', ascending = TRUE)
df_selection_sort[0:5]

##Выбираем кухню
Cuisine = st.selectbox(
        "Cuisine", df_new["cuisine"].value_counts().index
    )
df_selection2 = df_new[lambda x: x["cuisine"] == Cuisine]
df_selection2


