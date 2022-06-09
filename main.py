import streamlit as st
import wikipedia
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


search0 = df_selection['city'][0:1].values[0]
##list = search0.split(";")
##search0 = list[0]
search0 = wikipedia.search(search0)[0]
st.write(search0)
search0 = search0.replace(" ", "_")
url = 'https://en.wikipedia.org/wiki/' + search0
r = requests.get(url)
text = BeautifulSoup(r.text, 'html.parser')
for link in text("img"):
    a = link.get('src')
    if((a is None) == False):
        ans = a
        ind = 1
    if(ind == 1):
        break
st.write("https:"+ ans)




