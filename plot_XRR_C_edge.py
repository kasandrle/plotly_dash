# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

from os import listdir
from os.path import isfile, join, expanduser
import pickle
import numpy as np
from functools import reduce
import sys
home = expanduser("~")
sys.path.append(join(home,'Projects/helpfullscripts/'))
sys.path.append(join(home,'Projects/'))
sys.path.append(join(home,'Projects/XRR_workflows/calculate_n_k_from_xrr/'))
sys.path.append(join(home,'Projects/XRR_workflows/calculate_n_k_from_xrr/ZEP_XRR/'))
sys.path.append(home)
sys.path.append('../')
import xray_compounds as xc
import Divergence as div
import pint
unit = pint.UnitRegistry()

import matrixmethod.mm_numba as mm
import pandas as pd

# Samll helper function
import scipy.constants
c = scipy.constants.c #speed of light
h = scipy.constants.h #Planck
e = scipy.constants.e #elemetry charge
hc = h*c/e*1e9 #wavelenght in nm
#print(hc)
def eVnm_converter(value):
    #Planck's constant (6.6261 x 10-34 J*s) and c is the speed of light (2.9979 x 108 m/s)
    return hc/value


# Incorporate data #
aoi_deg = np.linspace(1e-3,40,100)
energies = np.array([280,285.2,288.6])

nk_ZP_exposed_all = pd.read_excel('./Optical constant ZEP.xlsx',sheet_name=0)
nk_ZP_unexposed_all = pd.read_excel('./Optical constant ZEP.xlsx',sheet_name=1)
f_n_ZP_exposed = scipy.interpolate.interp1d(nk_ZP_exposed_all['Energy (eV)'],1-nk_ZP_exposed_all['delta'], kind='linear')
f_k_ZP_exposed = scipy.interpolate.interp1d(nk_ZP_exposed_all['Energy (eV)'],nk_ZP_exposed_all['beta'], kind='linear')
f_n_ZP_unexposed = scipy.interpolate.interp1d(nk_ZP_unexposed_all['Energy (eV)'],1-nk_ZP_unexposed_all['delta'], kind='linear')
f_k_ZP_unexposed = scipy.interpolate.interp1d(nk_ZP_unexposed_all['Energy (eV)'],nk_ZP_unexposed_all['beta'], kind='linear')

nk_ZP_exposed = []
nk_ZP_unexposed = []
for energy in energies:
    nk_ZP_exposed.append(f_n_ZP_exposed(energy)+f_k_ZP_exposed(energy)*1j)
    nk_ZP_unexposed.append(f_n_ZP_unexposed(energy)+f_k_ZP_unexposed(energy)*1j)

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('XRR', className="text-primary text-center fs-3")
    ]),


    dbc.Row([
        html.Div('____________________________________________________', className="line")
    ]),

    dbc.Row([
       
        html.Div("Orders:"),
        dbc.RadioItems(options=[{"label": x, "value": x} for x in energies],
                       value=energies[0],
                       #inline=False,
                       id='radio-buttons-energy'),
        html.Div("thickness slider:"),
        dcc.Slider( 10, 40, step=None, id='fit_thickness--slider', value=10 ),
        html.Div("roughness slider:"),
        dcc.Slider( 0, 10, step=None, id='fit_rough--slider',value= 0 ),
        #html.Div("delta slider:"),
        #dcc.Slider( -1e-2, 1e-2, step=None, id='fit_delta--slider', value=0 ),
        #html.Div("beta slider:"),
        #dcc.Slider( 0, 1e-2, step=None, id='fit_beta--slider', value=1e-3 ),
        
    ]),

    dbc.Row([
  
        dbc.Col([
            dcc.Graph(figure={}, id='XRR')
        ], width=6),

    ]),

], fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='XRR', component_property='figure'),
    Input(component_id='radio-buttons-energy', component_property='value'),
    Input(component_id='fit_thickness--slider', component_property='value'),    
    Input(component_id='fit_rough--slider', component_property='value'),    
)
def update_graph(energy,thickness, rough):    
    #print(order)
    layer = np.array([thickness,1.2])
    rough = np.array([rough, 0.5, 0.5])

    nk_sub = np.conjugate(xc.refractive_index('Si',energy* unit.eV,density=2.33))
    nk_sub_oxid = np.conjugate(xc.refractive_index('SiO2',energy* unit.eV,density=2.0929729034348785))
    nk_ZP = f_n_ZP_unexposed(energy)+f_k_ZP_unexposed(energy)*1j
    wl = eVnm_converter(energy)
    n = np.array([1 + 0 * 1j,nk_ZP,nk_sub_oxid,nk_sub])
    aoi = np.deg2rad(aoi_deg)
    rm, tm = mm.reflec_and_trans(n, wl, aoi, layer, rough, 1)  # polarization (either 1 for s-polarization or 0 for p-polarization)
    rm = np.square(np.abs(np.asarray(rm)))

    #rm_p, tm = mm.reflec_and_trans(n, wl, aoi, layer, rough, 0)  # polarization (either 1 for s-polarization or 0 for p-polarization)
    #rm_p = np.square(np.abs(np.asarray(rm_p)))
    #print(df)
    fig = go.Figure()

    # Highlight the selected line
    fig.add_trace(
        go.Scatter(
            x=aoi_deg,
            y=rm,
            mode='lines',
            name='s pol',
            line=dict(color='red')
        )
    )
    
    fig.update_yaxes(type="log")
    fig.update_layout(
        title="XRR",
        xaxis_title="aoi [deg]",
        yaxis_title="Refelctivity",
    )

    return fig



# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=8053)
