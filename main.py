from email import utils
import streamlit as st
import pandas as pd
# from bokeh.plotting import figure
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
# from sklearn.neighbors import KernelDensity
import pickle
import utils


df = pd.read_csv("dfs.csv")


# utils.calc_density()
# df = pd.read_csv("df_alt.csv")
def p_idade(idade):
    kernel = gaussian_kde(df.idade)
    #kernel.set_bandwidth(kernel.factor / idade)
    df['density_idade'] = kernel.evaluate(df.idade)
    xs = np.linspace(0, 100, num=1000)
    fig = px.line(x=xs, y=kernel(xs))
    fig = px.area(x=xs, y=kernel(xs))
    fig.add_annotation(x=idade, y=kernel(idade)[0], text="A", showarrow=True, arrowhead=1)
    fig.update_layout(template="plotly_dark")
    fig.update_annotations()
    st.plotly_chart(fig, use_container_width=True)
    
def p_marca_uti():
    marca_uti = df['MARCA_UTI']
    marca_uti = marca_uti[marca_uti != 0]
    marca_uti = marca_uti[marca_uti != 1]
    
    marca_uti.reset_index()
    marca_uti = pd.DataFrame(marca_uti)
    marca_uti = marca_uti.value_counts(ascending=True)
    index = []
    for i in range(len(marca_uti.index)):
        index.append(marca_uti.index[i][0])
    
    #marca_uti['MARCA_UTI'] = f"{marca_uti['MARCA_UTI']}"
    fig = px.bar(x = marca_uti, y = index)
    fig.add_annotation(x=idade, text="P", showarrow=True, arrowhead=1)
    fig.update_xaxes(type='category')
    
    
    st.plotly_chart(fig)
    
def formatar_sexo(sexo):
    if sexo == 1:
        return("Masculino")
    elif sexo == 3:
        return("Feminino")
    else:
        return("Outro")
    
def ler_modelos():
    # with open("ESPEC_<_idade_PROC_SOLIC", 'rb') as file:
    #     lo = pickle.load(file)
        
    with open("VAL_SH_<_idade_SEXO.pickle", 'rb') as file2:
        fit_val_sh = pickle.load(file2)
        
    with open("VAL_SP_<_idade_SEXO.pickle", 'rb') as file2:
        fit_val_sp = pickle.load(file2)
        
    with open("VAL_UTI_<_idade_SEXO.pickle", 'rb') as file2:
        fit_val_uti = pickle.load(file2)
        
    return fit_val_sh, fit_val_sp, fit_val_uti

def paciente(idade, sexo):
    paciente = pd.Series({
        'idade': idade,
        'SEXO': sexo
    })
    print("Paciente:", paciente)
    return pd.DataFrame(paciente).transpose()
    
def prev_val_sh(idade, sexo):
    val = fit_val_sh.predict(paciente(idade, sexo))
    #print(val)
    return val

def prev_val_sp(idade, sexo):
    val = fit_val_sp.predict(paciente(idade, sexo))
    #print(val)
    return val

def prev_val_uti(idade, sexo):
    val = fit_val_uti.predict(paciente(idade, sexo))
    #print(val)
    return val


# """
# DASHBOARD
# """
 
fit_val_sh, fit_val_sp, fit_val_uti = ler_modelos()


with st.sidebar:
    st.write(""" 
         Dados do paciente
         """)
    idade = st.slider("Idade", min_value = min(df.idade), max_value= max(df.idade), step = 1.0)
    sexo = st.radio(label="Sexo", options=set(df.SEXO), format_func=formatar_sexo)

    
h_val_sh = np.round(prev_val_sh(idade, sexo), 2)
h_val_sp = np.round(prev_val_sp(idade, sexo), 2)
h_val_uti = np.round(prev_val_uti(idade, sexo), 2)
h_tot = h_val_sh + h_val_sp + h_val_uti
    
m1, m2, m3, m4 = st.columns(4)

st.write("Média de gastos com pacientes como este")
    
with st.container():
    with m1:
        st.metric("Valor de Total", value = h_tot)
    with m3:
        st.metric("Valor de serviços hospitalares", value = h_val_sh)
    with m2:
        st.metric("Valor de serviços profissionais", value = h_val_sp)
    with m4:    
        st.metric("Valor de UTI", value = h_val_uti)
    
with st.container():
    p_idade(idade)
    p_marca_uti()

