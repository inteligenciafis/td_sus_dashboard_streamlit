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