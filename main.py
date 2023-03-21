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
import folium
import json
from streamlit_folium import st_folium
import sklearn
from sklearn.linear_model import LinearRegression
import re
import networkx as nx
from networkx.algorithms import bipartite
from scipy.sparse import csc_matrix



#with st.echo(code_location='below'):

        def print_hello(name="World"):
                st.header(f" Привет, {name}! Дождись, пожалуйста, когда мальчишка в правом верхнем углу добежит (иногда это занимает некоторое время), только после начинай листать, чтобы не было проблем")
        st.write("Введите ваше имя")
        name = st.text_input("Your name", key="name", value="мой лучший друг")
        a = print_hello(name)
        st.markdown("В данном приложении вы узнаете немного о ресторанах Мишлен, сможете выбрать подходящий именно для Вас в любой точке мира, или же в любимой Москве. А также даже попробуете заказать пиццу.")

        df= pd.read_csv("df_23.csv")
        st.markdown("Давайте побольше узнаем о ценовых категориях в ресторанах Мишлен. Обратите внимание на обозначения среднего чека на человека.")
        st.markdown("Обозначения в долларах:")
        st.markdown("1 : 1-10")
        st.markdown("11 : 11-50")
        st.markdown("111 : 51-80")
        st.markdown("1111 : 81-120")
        st.markdown("11111 : 121-300")
        st.markdown("Итак, на следующем изображении вы можете увидеть, сколько ресторанов в каждой ценовой категории.")
        ### построение визуализации (вафля)
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
                'title': {'label': ' Распределение ценовых категорий у ресторанов Мишлен', 'loc': 'center', 'fontsize':26}
                },
        },
        rows=7,
        colors=colors,
        figsize=(16, 9)
        )
        ## Построение визуализации
        st.pyplot(figure)
        st.markdown("А какие же регионы наиболее дорогие и наоборот?")
        df2=df.sort_values(by=["price"])
        fig, ax = plt.subplots(figsize=(20,10))
        ax = sns.barplot(x="region", y="price", data=df2)
        plt.title('Distribution of regions by Michelen resturants price level', fontsize=30)
        st.pyplot(fig)

        ## Создание селект бокса (разные типы)
        st.header("Рестораны Мишлен")
        st.markdown("В данном разделе вы сможете подобрать ресторан Мишлен, который подходит именно вам.")
        st.markdown("Пожалуйста, выберите регион, в котором вы бы хотели испытать гастрономическое удовольствие.")
        Region = st.selectbox(
                "Region", df["region"].value_counts().index
        )
        df_selection = df[(df['region'] == Region)]
        st.markdown("Пожалуйста, выберите кухню, которые представлены в выбранном вами регионе.")
        Cuisine = st.selectbox(
                "Cuisine", df_selection["cuisine"].value_counts().index
        )
        df_selection = df[(df['region'] == Region) & (df['cuisine'] == Cuisine)]

        st.markdown("Пожалуйста, выберите средний чек на человека. Выбор будет доступен если в выбранном вами регионе с определенной кухней представлены рестораны разных ценовых категорий.")
        st.image('https://avatars.mds.yandex.net/i?id=630bf6a72ffd8e17c86e4908d3cddbc0-4230863-images-thumbs&n=13', width = 250)
        average_check = st.radio("Select option", df_selection['price'].unique())
        st.write("Average check:", average_check)
        df_selection = df_selection[df_selection["price"] ==average_check]
        st.markdown("В данной таблице представдены все рестораны, подходящие по параметрам, которые были выбраны ранее.")
        df_show = df_selection[["name", "region", "city", "price", "cuisine", "url"]]
        df_show

        st.markdown("Если в таблице представлены несколько ресторанов, можете выбрать любой из них.")
        restaurant = st.radio("Select option", df_selection['name'].unique())
        st.write("Restaurant:", restaurant)
        df_selection_2 = df_selection[df_selection["name"] ==restaurant]
        df_show = df_selection_2[["name", "region", "city", "price", "cuisine", "url"]]
        df_show
        st.markdown("Для того чтобы насладиться ужином в выбранном ресторане, вам необходимо будет совершить путешествие в другой город. Посмотрите в каком замечательном месте находится ваш ресторан.")
        st.markdown("P.S. (Данные описания и картинки города я получаю с помощью веб-скреппинга, однако не у всех ресторанов на сате Мишлен есть описание и картинка с достопримечательностями города. Попробуйте выбрать Taipei, Sweden, Austria Norway, Finland, Rio de Janeiro, Croatia, Hungary, Greece, Poland, Grech Republic, чтобы насладиться также описанием и фотографиями доспримечательностей)")
        st.markdown("P.S.2 (Так как программа находит картинки в онлайн режиме, за время после написания кода, сайт мог обновиться, поэтому если вдруг что-то не работает, попробуйте выбрать другой регион.)")
        
       
