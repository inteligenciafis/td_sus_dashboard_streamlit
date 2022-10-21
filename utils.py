#import streamlit as st
import pandas as pd
# from bokeh.plotting import figure
#import plotly.figure_factory as ff
#import plotly.express as px
import numpy as np
from scipy.stats import gaussian_kde
# from sklearn.neighbors import KernelDensity

def calc_density():
    df = pd.read_csv("dfs.csv")
    d = gaussian_kde(df.idade)
    df['density_idade'] = d.evaluate(df.idade)
    df.to_csv("df_alt.csv")

calc_density()
    

"""
Transformar dados

"""

def add_munic(df):
    df['key'] = df['MUNIC_RES']
    
    mun = pd.read_excel("RELATORIO_DTB_BRASIL_MUNICIPIO.xls")
    mun['key'] = mun['Região Geográfica Imediata']
    df = df.merge(mun[['key', 'Nome_Município']], on='key')
    return(df)

mun = pd.read_table('/home/vini/git/td_sus_dashboard_streamlit/br_municip.cnv', encoding='latin1', sep = '\t', header=None)
def read_cnv(path):
    pass


""" 

Normalização

"""

# EGA1
import statistics
import numpy as np

def z(df, var, valor):
    mean = np.mean(df[var])
    sd = np.std(df[var])
    return(statistics.NormalDist(mean, sd).zscore(valor))

