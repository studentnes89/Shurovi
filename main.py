import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv("data_one.csv")
df2=df.dropna()
df2['price'] = df2['price'].str.replace('$','1')


##Выбираем регион
df_new = df2[["name", "region", "city", "price", "cuisine", "url"]]
Region = st.selectbox(
        "Region", df_new["region"].value_counts().index
    )
df_selection = df_new[(df_new['region'] == Region)]

Cuisine = st.selectbox(
        "Cuisine", df_selection["cuisine"].value_counts().index
    )
df_selection = df_new[(df_new['region'] == Region) & (df_new['cuisine'] == Cuisine)]

selected_class = st.radio("Select Class", df_selection['price'].unique())
st.write("Selected Class:", selected_class)
st.write("Selected Class Type:", type(selected_class))
df_selection = df_selection[df_selection["price"] ==selected_class]
df_selection

##df_selection_sort = df_selection.sort_values('price', ascending = TRUE)
##df_selsction_sort


##Сортировать не раб







