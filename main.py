import streamlit as st
import wikipedia
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
from pywaffle import Waffle 

## вафля

df= pd.read_csv("data_one.csv")
df2=df.dropna()
df2['price'] = df2['price'].str.replace('$','1')
df_new = df2[["name", "region", "city", "price", "cuisine", "url"]]

df_vaf = df_new.groupby('price').size().reset_index(name='counts')
n_categories = df_vaf.shape[0]
colors = [plt.cm.inferno_r(i/float(n_categories)) for i in range(n_categories)]

figure = plt.figure(
    FigureClass=Waffle,
    plots={
        111: {
            'values': df_vaf['counts'],
            'labels': ["{0} ({1})".format(n[0], n[1]) for n in df_vaf[['price', 'counts']].itertuples()],
            'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12},
            'title': {'label': ' Распределение ценовых категорий у ресторанов Мишлен', 'loc': 'center', 'fontsize':18}
        },
    },
    rows=7,
    colors=colors,
    figsize=(16, 9)
)
st.pyplot(figure)


##Выбираем регион

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
selected_class = st.radio("Select Class", df_selection['name'].unique())
st.write("Selected Class:", selected_class)
st.write("Selected Class Type:", type(selected_class))
df_selection = df_selection[df_selection["name"] ==selected_class]
df_selection

cit = df_selection['city'][0:1].values[0]
cit = wikipedia.search(cit)[0]
st.write(cit)
cit = cit.replace(" ", "_")
ssilka = 'https://en.wikipedia.org/wiki/' + cit
r = requests.get(ssilka)
t = BeautifulSoup(r.text, 'html.parser')
for link in t("img"):
    a = link.get('src')
    if (a is None) == False:
        itog = a
        ind = 1
    if ind == 1:
        break
st.write("https:"+ itog)
url = "https:"+ itog
st.image(url)



pizza_df=pd.read_csv("pizza_df.csv")
pizza_df['company'] = pizza_df['company'].str.replace('A', "5")
pizza_df['company'] = pizza_df['company'].str.replace('B', "4")
pizza_df['company'] = pizza_df['company'].str.replace('C', "3")
pizza_df['company'] = pizza_df['company'].str.replace('D', "2")
pizza_df['company'] = pizza_df['company'].str.replace('E', "1")
pizza_df = pizza_df.rename (columns= {'price_rupiah': 'price'})
for i in range(len(pizza_df.index)):
    yacheika=pizza_df['price'][i:i+1].values[0]
    length=len(yacheika)
    new_yacheika=yacheika[2:(length-4)]
    pizza_df.loc[i,'price'] = new_yacheika
for i in range(len(pizza_df.index)):
    yach=pizza_df['diameter'][i:i+1].values[0]
    ln=len(yach)
    new_yach=yach[0:(ln-5)]
    pizza_df.loc[i,'diameter'] = new_yach
pizza_df['extra_sauce'] = pizza_df['extra_sauce'].str.replace('yes', "1")
pizza_df['extra_sauce'] = pizza_df['extra_sauce'].str.replace('no', "0")
pizza_df['extra_cheese'] = pizza_df['extra_cheese'].str.replace('yes', "1")
pizza_df['extra_cheese'] = pizza_df['extra_cheese'].str.replace('no', "0")
pizza_df['extra_mushrooms'] = pizza_df['extra_mushrooms'].str.replace("yes", "1")
pizza_df['extra_mushrooms'] = pizza_df['extra_mushrooms'].str.replace('no', "0")
pizza_df['company'] = pd.to_numeric(pizza_df['company'])
pizza_df['price'] = pd.to_numeric(pizza_df['price'])
pizza_df['extra_sauce'] = pd.to_numeric(pizza_df['extra_sauce'])
pizza_df['extra_cheese'] = pd.to_numeric(pizza_df['extra_cheese'])
pizza_df['extra_mushrooms'] = pd.to_numeric(pizza_df['extra_mushrooms'])
pizza_df
