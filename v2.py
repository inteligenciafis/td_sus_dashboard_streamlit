# from ast import Try
# from cProfile import label
# from email import utils
# from shutil import which
# from turtle import title
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
import graphviz as graphviz

import statistics
import numpy as np
import datetime as dt
import time

# """
# MODELOS DE REGRESSÃO
# """
with open("VAL_SH_<_idade_SEXO.pickle", 'rb') as file2:
        fit_val_sh = pickle.load(file2)
        
with open("VAL_SP_<_idade_SEXO.pickle", 'rb') as file2:
    fit_val_sp = pickle.load(file2)
    
with open("VAL_UTI_<_idade_SEXO.pickle", 'rb') as file2:
    fit_val_uti = pickle.load(file2)

with open("qt_diarias<_.pickle", 'rb') as file2:
    fit_diarias = pickle.load(file2)

# """ DF """
def p_idade(idade, idade_paciente):
    kernel = gaussian_kde(df.idade)
    #kernel.set_bandwidth(kernel.factor / idade)
    df['density_idade'] = kernel.evaluate(df.idade)
    xs = np.linspace(0, 100, num=1000)
    fig = px.line(x=xs, y=kernel(xs), labels={'x': "Idade", 'y': "Concentração"})
    fig.add_trace(go.Scatter(x=xs, y=kernel(xs), fill='tozeroy', name="Distribuição em todo Rio de Janeiro"))
    fig.add_trace(go.Scatter(x=pacientes.idade, y=kernel(pacientes.idade), mode="markers", name="Pacientes deste estabelecimento"))
    #fig = px.scatter(x=pacientes.idade, y=kernel(pacientes.idade))
    fig.add_annotation(
        x=idade_paciente[0], 
        y=kernel(idade_paciente)[0], 
        text=paciente.nome[0], 
        showarrow=True, 
        arrowhead=1,
        arrowsize=2,
        arrowwidth=2,
        arrowcolor="blue")
    fig.update_layout(template="plotly_dark")
    fig.update_annotations()
    st.plotly_chart(fig, use_container_width=True)

# """
# LEITURA DOS DADOS
# """

df = pd.read_csv("dfs.csv")
vars = ['idade', 'sexo', 'leito', 'procedimento', 'uti']


def format_espec(cod):
    espec = pd.read_csv("ESPEC.csv")
    return espec['ESPEC'][espec['cod'] == cod].values[0]

def format_cidsub(cod):
    cid = pd.read_csv("CID-10-SUBCATEGORIAS.CSV", sep = ";", encoding='latin1')
    return cid['DESCRICAO'][cid['SUBCAT'] == cod].values[0]

# cid["DESCRICAO"][cid['SUBCAT'] == "A040"].values[0]

pacientes = pd.read_csv("pacientes15cnome.csv")

idade = []
for i in range(len(pacientes)):
    pacientes.DATA_CMPT[i] = dt.datetime.strptime(pacientes.DATA_CMPT[i], "%Y-%m-%d")
    pacientes.NASC[i] = dt.datetime.strptime(pacientes.NASC[i], "%Y-%m-%d")
    idade.append(pacientes.DATA_CMPT[i] - pacientes.NASC[i])
    idade[i] = np.round(idade[i].days/365)
pacientes['idade'] = idade


# """
# STREAMLIT
#
# """
tab1, tab2= st.tabs(["Pacientes", "Hospitais"])

with tab1:
    # st.header("Pacientes")

    with st.sidebar:
        nome = st.selectbox("Escolha o paciente:", options=pacientes.nome)

    paciente = pacientes[pacientes.nome == nome].reset_index()

    try:
        paciente.MARCA_UTI
    except:
        paciente['MARCA_UTI'] = 00



    h_val_sh = np.round(fit_val_sh.predict(paciente[["idade", "SEXO", "ESPEC", "PROC_SOLIC", "MARCA_UTI"]]), 2)
    if h_val_sh[0] < 0:
        h_val_sh[0] = 0
    h_val_sp = np.round(fit_val_sp.predict(paciente[["idade", "SEXO", "ESPEC", "PROC_SOLIC", "MARCA_UTI"]]), 2)
    if h_val_sp[0] < 0:
        h_val_sp[0] = 0
    h_val_uti = np.round(fit_val_uti.predict(paciente[["idade", "SEXO", "ESPEC", "PROC_SOLIC", "MARCA_UTI"]]), 2)
    if h_val_uti[0] < 0:
        h_val_uti[0] = 0
    h_tot = np.round(h_val_sh + h_val_sp + h_val_uti, 2)

    c1, c2 = st.columns(2)

    with c1:
        sexo = '🧍‍♂️'
        if paciente.SEXO[0] != 3:
            sexo = 'Masculino🧍‍♂️' 
        else:
            sexo = 'Feminino 🧍‍♀️'

        #pacientes["idade"] = 
        st.title(paciente.nome[0])
        "(*Nome fictício*)"
        st.metric(label = "Sexo", value=sexo)
        st.metric(label="Idade", value=paciente['idade'])

    with c2:
        st.metric(label="Procedimento Solicitado", value=paciente['PROC_SOLIC'])
        st.metric(label="Procedimento Realizado", value=paciente['PROC_REA'])
        if paciente['PROC_SOLIC'][0] != paciente['PROC_REA'][0]:
            st.warning("Procedimento solicitado ainda não foi realizado!", icon="⚠️")

        if paciente['IND_VDRL'][0] == 0:
            vdrl = "✔️"
        else:
            vdrl = "❌"
        st.metric(label="VDRL", value=vdrl)

        # Tipo de Leito
        st.metric(label="Especialidade do leito:", value=format_espec(paciente['ESPEC'].values[0]))

        "Diagnóstico principal:"
        st.text(format_cidsub(paciente['DIAG_PRINC'][0]) + " (" + paciente['DIAG_PRINC'][0] + ")")
        # st.metric(label="CID10 Primário:", value=format_cidsub(paciente['DIAG_PRINC'][0]))
        # st.metric(label="CID10 Secundário:", value=format_cidsub(paciente['DIAG_SECUN'][0]))

    "---"

    h_qt_diarias = np.round(fit_diarias.predict(paciente[["idade", "SEXO", "ESPEC", "PROC_SOLIC", "MARCA_UTI"]]), 2)
    st.metric("Diárias", value = paciente['QT_DIARIAS'], delta=np.round(paciente['QT_DIARIAS'][0] - h_qt_diarias[0], 1), delta_color='inverse')
    "*Comparação de acordo com nossa estimativa para pacientes semelhantes.*"

    "---"

    v1, v2, v3, v4 = st.columns(4)
    with st.container():
        with v1:
            st.metric(label = "Serviços Hospitalares:", value=f'R${paciente.VAL_SH[0]}', delta=np.round(paciente.VAL_SH[0]-h_val_sh[0], 2), delta_color='inverse')
        with v2:
            st.metric(label = "Serviços Prestados por Terceiros:", value=f'R${paciente.VAL_SP[0]}', delta=np.round(paciente.VAL_SP[0]-h_val_sp[0],2), delta_color='inverse')
        with v3:
            st.metric(label = "Valor gasto com UTI:", value=f'R${paciente.VAL_UTI[0]}', delta=np.round(paciente.VAL_UTI[0]-h_val_uti[0],2), delta_color='inverse')
        with v4:
            st.metric(label = "Total:", value=f'R${paciente.VAL_TOT[0]}', delta=np.round(paciente.VAL_TOT[0]-h_tot[0],2), delta_color='inverse')
        "*Comparações de acordo com nossa estimativa para pacientes semelhantes.*"


    p_idade(df.idade, paciente.idade)