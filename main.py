from ast import Try
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

import statistics
import numpy as np
import time

def z(df, var, valor):
    mean = np.mean(df[var])
    sd = np.std(df[var])
    return(statistics.NormalDist(mean, sd).zscore(valor))

df = pd.read_csv("dfs.csv")
vars = ['idade', 'sexo', 'leito', 'procedimento', 'uti']
# utils.calc_density()
# df = pd.read_csv("df_alt.csv")

def p_radar(df, idade, sexo, leito, procedimento, uti, h_val_sh, h_val_sp, h_val_uti):
    # print(fit_val_sh.predict(paciente(idade, sexo, leito)))
    
    
    r = [z(df, "idade", idade), 
         z(df, "VAL_SH", h_val_sh[0]), 
         z(df, "VAL_SP", h_val_sp[0]), 
         z(df, "VAL_UTI", h_val_uti[0]),]
    
    theta = ["Idade", "R$ SH", "R$ SP", "R$ UTI"]
    
    fig = px.line_polar(r=r, theta=theta, line_close=True)
    #r2 = [z(df, 'idade', idade), z(df, 'SEXO', sexo), z(df, 'ESPEC', leito)]
    #theta2 = ["Idade", "Sexo", "Leito"]
    
    #fig.add_trace(go.Scatterpolar(r=r2, theta=theta2, fill='toself'))
    fig.update_layout(
        
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-5, 5]
            )
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# def p_radar(idade, sexo, leito, df):
    
#     fig = go.Figure(data=go.Scatterpolar(
#         
#         fill='toself',
#     ))
    
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#             ),
#         ),
#         showlegend=False
#     )
#     fig.update_polars()
#     st.plotly_chart(fig, use_container_width=True)

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
    
    
def formatar_leito(leito):
    if leito == 1:
        return("Cirúrgico")
    elif leito == 2:
        return("Obstétricos")
    elif leito == 3:
        return("Clínico")
    elif leito == 4:
        return("Crônicos")
    elif leito == 5:
        return("Psiquiatria")
    elif leito == 6:
        return("Pneumologia")
    elif leito == 7:
        return("Pediátricos")
    elif leito == 8:
        return("Reabilitação")
    elif leito == 9:
        return("Dia/Cirúrgicos")
    elif leito == 10:
        return("Dia/AIDS")
    elif leito == 14:
        return("Dia/Saúde Mental")
    elif leito == 87:
        return("Saúde Mental (Clínico)")
            
def formatar_uti(uti):
    if uti == 0:
        return("Não utilizou UTI")
    elif uti == 1:
        return("Mais de um tipo de UTI")
    elif uti == 99:
        return("UTI Doador")
    elif uti == 74:
        return("UTI Adulto - Tipo I")
    elif uti == 75:
        return("UTI Adulto - Tipo II")
    elif uti == 76:
        return("UTI Adulto - Tipo III")
    elif uti == 77:
        return("UTI Infantil - Tipo I")
    elif uti == 78:
        return("UTI Infantil - Tipo II")
    elif uti == 79:
        return("UTI Infantil - Tipo III")
    elif uti == 80:
        return("UTI Neonatal - Tipo I")
    elif uti == 81:
        return("UTI Neonatal - Tipo II")
    elif uti == 82:
        return("UTI Neonatal - Tipo III")
    elif uti == 51:
        return("UTI adulto - tipo II COVID 19")
    elif uti == 83:
        return("UTI de queimados")
    elif uti == 85:
        return("UTI coronariana tipo II - UCO tipo II")
    elif uti == 52:
        return("UTI pediátrica - tipo II COVID 19")
    
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

def paciente(idade, sexo, leito, procedimento, uti):
    paciente = pd.Series({
        'idade': idade,
        'SEXO': sexo,
        'ESPEC': leito,
        'PROC_SOLIC': procedimento,
        'MARCA_UTI':uti,
    })
    
    print("Paciente:", paciente)
    return pd.DataFrame(paciente).transpose()
    
def prev_val_sh(idade, sexo, leito, procedimento, uti):
    val = fit_val_sh.predict(paciente(idade, sexo, leito, procedimento, uti))
    #print(val)
    return val

def prev_val_sp(idade, sexo, leito, procedimento, uti):
    val = fit_val_sp.predict(paciente(idade, sexo, leito, procedimento, uti))
    #print(val)
    return val

def prev_val_uti(idade, sexo, leito, procedimento, uti):
    val = fit_val_uti.predict(paciente(idade, sexo, leito, procedimento, uti))
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
    
    leito = st.radio(label="Tipo de leito", options=set(df.ESPEC), format_func=formatar_leito)
    
    procedimento = st.number_input(label="Procedimento solicitado ou realizado", min_value=0, max_value=99999999, value=99999999,
                                   help="Código do procedimento (número apenas)")
    
    uti = st.radio(label="UTI utilizada", options=set(df.MARCA_UTI), format_func=formatar_uti)
    datain = st.date_input("Data da admissão")
    # cid = st.selectbox(label="CID10 Primário", options=set(df.CID_NOTIF))
    
    st.write("\n \n \n")

    
h_val_sh = np.round(prev_val_sh(idade, sexo, leito, procedimento, uti), 2)
h_val_sp = np.round(prev_val_sp(idade, sexo, leito, procedimento, uti), 2)
h_val_uti = np.round(prev_val_uti(idade, sexo, leito, procedimento, uti), 2)
h_tot = np.round(h_val_sh + h_val_sp + h_val_uti, 2)
    
with st.container():
    "Seu paciente:"
    c1, c2, c3 = st.columns(3)
    with c1:
        if sexo == 1:
            st.metric("Sexo", "🧍‍♂️(H)")
        if sexo == 3:
            st.metric("Sexo", "🧍‍♀️(M)")
    with c2:
        st.metric("Idade:", idade)
    with c3:
        st.metric(label="Procedimento solicitado:", value=int(procedimento))

with st.container():
    "Seu paciente:"
    a1, a2, a3 = st.columns(3)
    with a1:
        pass #st.metric("Dias de internação", diarias)
    with a2:
        st.metric("Idade:", idade)
    with a3:
        st.metric(label="Procedimento solicitado:", value=int(procedimento))


m1, m2, m3, m4 = st.columns(4)

with st.container():
    
    st.write("Média de gastos com pacientes como este")
    
    with m1:
        st.metric("Valor total esperado", value = f"R$ {h_tot[0]}")
    with m3:
        st.metric("Valor esperado de serviços hospitalares", value = f"R$ {h_val_sh[0]}")
    with m2:
        st.metric("Valor esperado de serviços profissionais", value = f"R$ {h_val_sp[0]}")
    with m4:    
        st.metric("Valor esperado de UTI", value = f"R$ {h_val_uti[0]}")
    
with st.container():
    mm1, mm2 = st.columns(2)
    with mm1:
        p_radar(df, idade, sexo, leito, procedimento, uti, h_val_sh, h_val_sp, h_val_uti)
    with mm2: 
        p_idade(idade)
        p_marca_uti()
