import streamlit as st
  import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 import seaborn as sns
 from mpl_toolkits.mplot3d import Axes3D
import folium
from streamlit_folium import st_folium
import json


st.header("Michelin restaurants")
df= pd.read_csv("data_one.csv")
