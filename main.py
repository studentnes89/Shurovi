import streamlit as st
import wikipedia
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import requests
from pywaffle import Waffle 
import geopandas as gpd
import folium
import json
from geopandas.tools import geocode
from streamlit_folium import st_folium

## вафля

df= pd.read_csv("df_23.csv")


df_vaf = df.groupby('price').size().reset_index(name='counts')
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
        "Region", df["region"].value_counts().index
    )
df_selection = df[(df['region'] == Region)]

Cuisine = st.selectbox(
        "Cuisine", df_selection["cuisine"].value_counts().index
    )
df_selection = df[(df['region'] == Region) & (df['cuisine'] == Cuisine)]

selected_class = st.radio("Select Class", df_selection['price'].unique())
st.write("Selected Class:", selected_class)
st.write("Selected Class Type:", type(selected_class))
df_selection = df_selection[df_selection["price"] ==selected_class]
df_show = df_selection[["name", "region", "city", "price", "cuisine", "url"]]
df_show

selected_class = st.radio("Select Class", df_selection['name'].unique())
st.write("Selected Class:", selected_class)
st.write("Selected Class Type:", type(selected_class))
df_selection = df_selection[df_selection["name"] ==selected_class]
df_show = df_selection[["name", "region", "city", "price", "cuisine", "url"]]
df_show
discrp=df_selection['description'][0:1].values[0]
discrp

cit = df_selection['city'][0:1].values[0]
cit = wikipedia.search(cit)[0]
st.write(cit)
city = cit.replace(" ", "_")
ssilka = 'https://en.wikipedia.org/wiki/' + city
r = requests.get(ssilka)
t = BeautifulSoup(r.text, 'html.parser')
for link in t("img"):
    a = link.get('src')
    if (a is None) == False:
        itog = a
        ind = 1
    if ind == 1:
        break
url = "https:"+ itog
st.image(url)
##Карта
#loc = 'cit'
#location = geocode(loc, provider="nominatim" , user_agent = 'my_request')
#point = location.geometry.iloc[0] 

#data= pd.DataFrame({"longitude":[point.x], "latitude":[point.y]})

#mapit = folium.Map( location=[0, 0], zoom_start=1 ) 
#for lat , lon in zip(data.latitude , data.longitude): 
#        folium.Marker( location=[ lat,lon ], fill_color='#43d9de', radius=8 ).add_to( mapit ) 
#st_data = st_folium(mapit, width = 725)
#st_data
###Карта

#lat = df_selection['latitude']
#lon = df_selection['longitude']
#elevation = df_selection['name']
#map = folium.Map(location=[lat, lon], zoom_start = 5)
#folium.TileLayer('cartodbpositron').add_to(map)
#for lat, lon, elevation in zip(lat, lon, elevation):
#    folium.Marker(location=[lat, lon], popup=str(elevation), icon=folium.Icon(color = 'pink')).add_to(map)
#st_data=st_folium(map, width=900)

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


#fig = plt.figure(figsize=(50, 20), dpi= 80)
#sns.heatmap(pizza_df.corr(), xticklabels=pizza_df.corr().columns, yticklabels=pizza_df.corr().columns, cmap='RdYlGn', center=0, annot=True)
#plt.title('Correlogram of pizza', fontsize=60)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#st.pyplot(fig)

fig = pizza_df.plt.scatter(x="diameter", y="price")
st.pyplot(fig)
