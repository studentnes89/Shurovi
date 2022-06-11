import streamlit as st
import wikipedia
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from bs4 import BeautifulSoup
import requests
from pywaffle import Waffle 
import geopandas as gpd
import folium
import json
from geopandas.tools import geocode
from streamlit_folium import st_folium
import sklearn
from sklearn.linear_model import LinearRegression
import re

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

average_check = st.radio("Select option", df_selection['price'].unique())
st.write("Average check:", average_check)
df_selection = df_selection[df_selection["price"] ==average_check]
df_show = df_selection[["name", "region", "city", "price", "cuisine", "url"]]
df_show

restaurant = st.radio("Select option", df_selection['name'].unique())
st.write("Restaurant:", restaurant)
df_selection_2 = df_selection[df_selection["name"] ==restaurant]
df_show = df_selection_2[["name", "region", "city", "price", "cuisine", "url"]]
df_show
discrp=df_selection_2['description'][0:1].values[0]
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
lat = df_selection['latitude']
lon = df_selection['longitude']
name = df_selection['name']
name_2 = df_selection_2['name']
lat_2 = df_selection_2['latitude']
lon_2 = df_selection_2['longitude']
map = folium.Map(location=[lat_2, lon_2], zoom_start = 9)
folium.TileLayer('cartodbpositron').add_to(map)
for lat, lon, name in zip(lat, lon, name):
    folium.Marker(location=[lat, lon], tooltip=str(name), icon=folium.Icon(color = 'gray' ), legend_name="Ресторан").add_to(map)
for lat_2, lon_2, name_2 in zip(lat_2, lon_2, name_2):
    folium.Marker(location=[lat_2, lon_2], tooltip=str(name_2), icon=folium.Icon(color = 'pink' ), legend_name="Ресторан").add_to(map)

st_data=st_folium(map, width=900)

f = open("textik.txt")
textik = f.read()
Moscow_restaurants = re.findall("\d.«([^»]+)»", textik)
st.write(Moscow_restaurants)
f.close()


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

#sns.set_style("white")
#gridobj = sns.lmplot(x="diameter", y="price", data=pizza_df)
#st.pyplot(gridobj)

pizza_df=pizza_df[["company", "price", "diameter", "extra_sauce", "extra_cheese", "extra_mushrooms"]]
pizza_df["price"] = pizza_df.price.mul(3)

Company_raiting = st.selectbox(
        "Company", (5, 4, 3, 2, 1)
    )

Diameter = st.columns(2)
diameter = Diameter[0].number_input("Diameter", value = 20)

Extra_sauce = st.expander("Optional addings", True)
extra_sauce = Extra_sauce.slider(
    "Extra Sauce",
    min_value = 0.0,
    max_value = 10.0
)
Extra_cheese = st.expander("Optional addings", True)
extra_cheese = Extra_cheese.slider(
    "Extra cheese",
    min_value = 0.0,
    max_value = 10.0
)

Extra_mushrooms = st.expander("Optional addings", True)
extra_mushrooms = Extra_mushrooms.slider(
    "Extra mushrooms",
    min_value = 0.0,
    max_value = 10.0
)

model = LinearRegression()
model.fit(pizza_df.drop(columns=["price"]), pizza_df["price"])
price = model.intercept_ + Company_raiting*model.coef_[0] + diameter*model.coef_[1] + extra_sauce*model.coef_[2] +extra_cheese*model.coef_[3] + extra_mushrooms*model.coef_[4]

url_sir = "http://tuimazy-sushi.ru/uploads/newNew/0.jpg"
url_gribi = "https://sakura-rolls31.ru/image/cache/catalog/gubkin/grib-1000x700.jpg"
url_sous = "https://eatwell101.club/wp-content/uploads/2019/09/best-marinara-sauce-for-pizza-inspirational-pizza-sauce-vs-marinara-surprising-similarities-and-of-best-marinara-sauce-for-pizza.jpg"
##Доп картинки
if (extra_cheese >0) and (extra_sauce == 0) and (extra_mushrooms == 0):
    st.image(url_sir)
if (extra_cheese >0) and (extra_sauce == 0) and (extra_mushrooms > 0):
    st.image(url_gribi)
if (extra_cheese >0) and (extra_sauce > 0) and (extra_mushrooms > 0):
    st.image(url_gribi)
if (extra_cheese == 0) and (extra_sauce == 0) and (extra_mushrooms > 0):
    st.image(url_gribi)
if (extra_cheese >0) and (extra_sauce > 0) and (extra_mushrooms == 0):
    st.image(url_sir)
if (extra_cheese == 0) and (extra_sauce > 0) and (extra_mushrooms == 0):
    st.image(url_sous)
st.write("Price:", price)


