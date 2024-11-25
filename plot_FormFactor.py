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
home = expanduser('~')
sys.path.append(join(home,'github/XRR_workflows/calculate_n_k_from_xrr'))
sys.path.append(join(home,'Projects/XRR_workflows/calculate_n_k_from_xrr'))
import xray_compounds as xc
import pint
unit = pint.UnitRegistry()

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
def sld_converter(value,energy):
    lam = eVnm_converter(energy)#*10
    return np.square(lam)/(2*np.pi)*value#/1e6
def multi_gaussian(x,*params):
    y = np.zeros_like(x)   
    for i in range(0, len(params), 3):
        a = params[i]
        mu = params[i + 1]
        sigma = params[i + 2]
        y += a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return y


def calc_Div_angles(div, dtheta, theta):
    # Calculate sigma for divergence
    sigma = (div / 1000.0 * 180.0 / np.pi) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    
    # Step size dependent on sigma
    dschritt = np.ceil(3.0 * sigma / 5.0 / dtheta) * dtheta
    div_calc_arr = np.linspace(-5, 5, 11) * dschritt
    
    min_theta = theta.min()
    
    # Initialize lists to accumulate results
    theta_for_div_sim = []
    positionsarray = []
    
    for i in range(len(theta)):
        temp_positions = []
        for j, offset in enumerate(div_calc_arr):
            new_theta = theta[i] + offset
            if new_theta >= min_theta:
                theta_for_div_sim.append(new_theta)
                if j == 5:
                    temp_positions.append(len(theta_for_div_sim) - 1)
        if temp_positions:
            positionsarray.append(temp_positions[0])
    
    return {
        'new_theta_for_calc': np.array(theta_for_div_sim),
        'original_positions': np.array(positionsarray)
    }

def calc_Divergenz(div, dtheta, positionsarray, theta, int_array):
    # Calculate sigma and divergence array
    sigma = (div / 1000.0 * 180.0 / np.pi) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dschritt = np.ceil(3.0 * sigma / 5.0 / dtheta) * dtheta
    div_calc_arr = np.linspace(-5, 5, 11) * dschritt
    div_erg_arr = np.exp(-0.5 * (div_calc_arr / sigma) ** 2)
    
    min_theta = theta.min()
    is_2d = np.ndim(int_array) == 2
    output_shape = (len(positionsarray), int_array.shape[1]) if is_2d else len(positionsarray)
    output = np.zeros(output_shape)
    
    for j, pos in enumerate(positionsarray):
        div_erg_arr_total = 0.0
        for i, weight in enumerate(div_erg_arr):
            offset_pos = pos + (i - 5)
            if theta[pos] + div_calc_arr[i] >= min_theta:
                if is_2d:
                    output[j, :] += int_array[offset_pos, :] * weight
                else:
                    output[j] += int_array[offset_pos] * weight
                div_erg_arr_total += weight
        if div_erg_arr_total > 0:
            output[j] /= div_erg_arr_total

    return output

def Intensity_q_max(q,width,thickness,n_medium,n_shell,n_core,N):
    """
    N - is the number of grating lines
    d - is the pitch or the distance between two lines
    w - width of the core
    t - thickness of the shell
    q - q space values
    """
    I_core_shell = np.abs((n_core-n_shell)*width*np.sinc(q*width/2/ np.pi))
    I_shell_medium = np.abs((n_shell-n_medium)*(width+2*thickness)*np.sinc(q*(width+2*thickness)/2 / np.pi)*np.exp(q*width*1j))
    return np.square((I_core_shell+I_shell_medium))*N #* np.square(N) #np.square(np.sin(N*q*pitch/2))/np.square(np.sin(q*pitch/2))

def Intensity_q_max_filter(q,width,thickness,n_medium,n_shell,n_core,N, divergenz):
    """
    N - is the number of grating lines
    d - is the pitch or the distance between two lines
    w - width of the core
    t - thickness of the shell
    q - q space values
    """
    DivAngles = calc_Div_angles(divergenz, 0.00001, q)
    q_ = DivAngles['new_theta_for_calc']
    positionsarray = DivAngles['original_positions']
    I_core_shell = np.abs((n_core-n_shell)*width*np.sinc(q_*width/2/ np.pi))
    I_shell_medium = np.abs((n_shell-n_medium)*(width+2*thickness)*np.sinc(q_*(width+2*thickness)/2 / np.pi)*np.exp(q_*width*1j))
    I = np.square(np.abs(I_core_shell+I_shell_medium))* N
    rm = calc_Divergenz(divergenz, q_[1] - q_[0], positionsarray, q_, I)
    return rm #np.square(np.sin(N*q*pitch/2))/np.square(np.sin(q*pitch/2))


def Intensity_q_max_filter_core(q,width,n_medium,n_core,N, divergenz):
    """
    N - is the number of grating lines
    d - is the pitch or the distance between two lines
    w - width of the core
    t - thickness of the shell
    q - q space values
    """
    DivAngles = calc_Div_angles(divergenz, 0.00001, q)
    q_ = DivAngles['new_theta_for_calc']
    positionsarray = DivAngles['original_positions']
    I_core_shell = np.abs((n_core-n_medium)*width*np.sinc(q_*width/2/ np.pi))
    I = np.square(np.abs(I_core_shell))* N
    rm = calc_Divergenz(divergenz, q_[1] - q_[0], positionsarray, q_, I)
    return rm #np.square(np.sin(N*q*pitch/2))/np.square(np.sin(q*pitch/2))

def Intensity_q_max_core(q,width,n_medium,n_core,N):
    """
    N - is the number of grating lines
    d - is the pitch or the distance between two lines
    w - width of the core
    t - thickness of the shell
    q - q space values
    """
    I_core_shell = np.abs((n_core-n_medium)*width*np.sinc(q*width/2/ np.pi))
    I = np.square(np.abs(I_core_shell))* N
    return I #np.square(np.sin(N*q*pitch/2))/np.square(np.sin(q*pitch/2))

# Incorporate data #
N= 1
q = np.linspace(0,0.3,100)



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
        html.Div("width slider:"), 
        dcc.Slider( 50, 150, step=None, id='fit_width--slider', value=100 ),
        html.Div("thickness slider:"),
        dcc.Slider( 0, 45, step=None, id='fit_thickness--slider', value=10 ),
        html.Div("n_medium slider:"),
        dcc.Slider( 0, 1, step=None, id='fit_n_medium--slider',value= 0 ),
        html.Div("n_shell slider:"),
        dcc.Slider( 0, 1, step=None, id='fit_n_shell_delta--slider', value=1 ),
        html.Div("n_core slider:"),
        dcc.Slider( 0, 1, step=None, id='fit_n_core--slider', value=0.5 ),
        html.Div("divergenz slider:"),
        dcc.Slider( 0, 1, step=None, id='fit_divergenz--slider', value=0.5 ),
        html.Div("pitch slider:"),
        dcc.Slider( 190, 210, step=None, id='fit_pitch--slider', value=200 ),
        
    ]),

    dbc.Row([
  
        dbc.Col([
            dcc.Graph(figure={}, id='fit_order-scan')
        ], width=6),

    ]),

], fluid=True)

# Add controls to build the interaction


@callback(
    Output(component_id='fit_order-scan', component_property='figure'),
    Input(component_id='fit_width--slider', component_property='value'),
    Input(component_id='fit_thickness--slider', component_property='value'),
    Input(component_id='fit_n_medium--slider', component_property='value'), 
    Input(component_id='fit_n_shell_delta--slider', component_property='value'),   
    Input(component_id='fit_n_core--slider', component_property='value'),   
    Input(component_id='fit_divergenz--slider', component_property='value'),
    Input(component_id='fit_pitch--slider', component_property='value'),   
)
def update_graph_fit(width,thickness,n_medium,n_shell,n_core, divergenz,pitch): 

    orders_ = np.asarray([2,3,4,5,6,7,8,9])
    q_max = np.empty(np.shape(orders_))
    for i,order in enumerate(orders_):
        q_max[i] = 2*np.pi/pitch*order

    I_q_core = Intensity_q_max_core(q,width,n_medium,n_core,N)
    I_q_core_div = Intensity_q_max_filter_core(q,width,n_medium,n_core,N, divergenz)
    I_q = Intensity_q_max(q,width,thickness,n_medium,n_shell,n_core,N)
    I_q_div = Intensity_q_max_filter(q,width,thickness,n_medium,n_shell,n_core,N, divergenz)

    I_qmax_core = Intensity_q_max_core(q_max,width,n_medium,n_core,N)
    I_qmax_core_div = Intensity_q_max_filter_core(q_max,width,n_medium,n_core,N, divergenz)
    I_qmax = Intensity_q_max(q_max,width,thickness,n_medium,n_shell,n_core,N)
    I_qmax_div = Intensity_q_max_filter(q_max,width,thickness,n_medium,n_shell,n_core,N, divergenz)

    I_q_list = [I_q_core,I_q_core_div,I_q,I_q_div]
    name_q = ['core','core div','shell core','shell div']
    I_qmax_list = [I_qmax_core,I_qmax_core_div,I_qmax,I_qmax_div]


    fig = go.Figure()
    for i,entry in enumerate(I_q_list):
        fig.add_trace(
            go.Scatter(
                x=q,
                y=entry,
                mode='lines',
                name = name_q[i]
            )
        )
    for entry in I_qmax_list:
        fig.add_trace(
            go.Scatter(
                x=q_max,
                y=entry,
                mode='markers',
                line=dict(color='black')
            )
        )
    fig.update_yaxes(type="log")
    fig.update_layout(
        title="Fromfactor",
        xaxis_title="q / nm",
        yaxis_title="Fromfactor",
    )

    
    return fig


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True,port=8053)
