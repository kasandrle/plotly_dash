# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
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

def Intensity_q_max(q,width,thickness,n_medium,n_shell,n_core,N,pitch):
    """
    N - is the number of grating lines
    d - is the pitch or the distance between two lines
    w - width of the core
    t - thickness of the shell
    q - q space values
    """
    n_medium = np.square((n_medium))
    n_shell = np.square((n_shell))
    n_core = np.square((n_core))
    I_core_shell = np.abs((n_core-n_shell)*width*np.sinc(q*width/2))
    I_shell_medium = np.abs((n_shell-n_medium)*(width+2*thickness)*np.sinc(q*(width+2*thickness)/2)*np.exp(q*width*1j))
    return np.square((I_core_shell+I_shell_medium))*N #* np.square(N) #np.square(np.sin(N*q*pitch/2))/np.square(np.sin(q*pitch/2))

def Intensity_q_max_filter(q,width,thickness,n_medium,n_shell,n_core,N,pitch, divergenz):
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
    n_medium = np.square(n_medium)
    n_shell = np.square(n_shell)
    n_core = np.square(n_core)
    I_core_shell = np.abs((n_core-n_shell)*width*np.sinc(q_*width/2))
    I_shell_medium = np.abs((n_shell-n_medium)*(width+2*thickness)*np.sinc(q_*(width+2*thickness)/2)*np.exp(q_*width*1j))
    I = np.square(np.abs(I_core_shell+I_shell_medium))* N
    rm = calc_Divergenz(divergenz, q_[1] - q_[0], positionsarray, q_, I)
    return rm #np.square(np.sin(N*q*pitch/2))/np.square(np.sin(q*pitch/2))

# Incorporate data #
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
path = '/home/kas/Projects/Ni_project/stand_alone_paper/20240524_YaoYang/fits_files/'
#path = '/home/kas/Projects/Ni_project/exp_data_paper/20240524_YaoYang/'
onlyfiles_keys = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('p9.csv') and f.startswith('')]
exp_data_all = []
for file in onlyfiles_keys:#[0::5]:
    #print(file)
    file = join(path, file)
    df = pd.read_csv(file)
    
    exp_data_all.append(df)
orders = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']
orders_ = [2,3,4,5,6,7,8,9]
orders_ = np.asarray(orders_)
q_max = np.empty(np.shape(orders))
for i,order in enumerate(orders_):
    q_max[i] = 2*np.pi/200*order

I0_Ni = pd.read_csv('/home/kas/Projects/Ni_project/XRR/I0_Ni_edge.csv')
f_I0_Ni = scipy.interpolate.interp1d(I0_Ni['energy'],I0_Ni['I0'], kind='linear')

nk_Ni_all = pd.read_excel('/home/kas/Projects/Ni_project/Ni reference spectra-optical constant-05172024.xlsx',sheet_name=1)
nk_NiO2_all = pd.read_excel('/home/kas/Projects/Ni_project/Ni reference spectra-optical constant-05172024.xlsx',sheet_name=2)
nk_NiO3_all = pd.read_excel('/home/kas/Projects/Ni_project/Ni reference spectra-optical constant-05172024.xlsx',sheet_name=8)
f_n_Ni = scipy.interpolate.interp1d(nk_Ni_all['energy'],1-nk_Ni_all['delta'], kind='linear')
f_k_Ni = scipy.interpolate.interp1d(nk_Ni_all['energy'],nk_Ni_all['beta'], kind='linear')
f_n_NiO2 = scipy.interpolate.interp1d(nk_NiO2_all['energy'],1-nk_NiO2_all['delta'], kind='linear')
f_k_NiO2 = scipy.interpolate.interp1d(nk_NiO2_all['energy'],nk_NiO2_all['beta'], kind='linear')
f_n_NiO3 = scipy.interpolate.interp1d(nk_NiO3_all['energy'],1-nk_NiO3_all['delta'], kind='linear')
f_k_NiO3 = scipy.interpolate.interp1d(nk_NiO3_all['energy'],nk_NiO3_all['beta'], kind='linear')
real_energy = np.asarray(exp_data_all[0]['energy_list'])[2:-5]
nk_H2O = (xc.refractive_index('H2O', real_energy * unit.eV, density=1))
SLD_H2O = sld_converter(nk_H2O,real_energy) 



# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('Ni exp data', className="text-primary text-center fs-3")
    ]),

    dbc.Row([
        dbc.Checklist(options=[{"label": x, "value": i} for i,x in enumerate(onlyfiles_keys)],
                       value=[0],
                       inline=True,
                       id='radio-buttons-final')
    ]),

    dbc.Row([
        html.Div('____________________________________________________', className="line")
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Checklist(options=[{"label": x, "value": x} for x in orders_],
                       value=orders_[0],
                       inline=False,
                       id='radio-buttons-order')
        ], width=2),

        dbc.Col([
            dcc.Graph(figure={}, id='energy-scan')
        ], width=6),

        dbc.Col([
        ], width=2),

        dbc.Col([
            dcc.Slider(
                exp_data_all[0]['energy_list'].min(),
                exp_data_all[0]['energy_list'].max(),
                step=None,
                id='crossfilter-energy--slider',
                value=exp_data_all[0]['energy_list'].max(),
                marks={energy: str('') for energy in exp_data_all[0]['energy_list'].unique()}
            ),
            dcc.Graph(figure={}, id='order-scan')
        ], width=6),


    ]),

    dbc.Row([
        html.Div('____________________________________________________', className="line")
    ]),

    dbc.Row([
        html.Div("Width slider:"),
        dcc.Slider( 50, 135, step=None, id='fit_width--slider', value=55 ),
        html.Div("thickness slider:"),
        dcc.Slider( 0, 45, step=None, id='fit_height--slider', value=10 ),
        html.Div("E Ni offset slider:"),
        dcc.Slider( -1, 1, step=None, id='fit_E_Ni--slider', value=0 ),
        html.Div("E NiO offset slider:"),
        dcc.Slider( -1, 1, step=None, id='fit_E_NiO--slider', value=0 ),
        html.Div("Scale slider:"),
        dcc.Slider( 0, 100, step=None, id='fit_N--slider',value= 55 ),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.RadioItems(options=[{"label": x, "value": x} for x in orders],
                       value=orders[0],
                       inline=False,
                       id='fit_radio-buttons-order')
        ], width=2),

        dbc.Col([
            dcc.Graph(figure={}, id='fit_energy-scan')
        ], width=6),

   

        dbc.Col([
            html.Div("Energy slider:"),
            dcc.Slider(
                real_energy.min(),
                real_energy.max(),
                step=None,
                id='fit_crossfilter-energy--slider',
                value=real_energy.max(),
                marks={energy: str('') for energy in real_energy}
            ),
            dcc.Graph(figure={}, id='fit_order-scan')
        ], width=6),


    ]),

], fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='energy-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='radio-buttons-order', component_property='value'),    
)
def update_graph(entry,order):    
    #print(order)
    df_list = []
    names = []
    for i in entry:
        for o in order:
            x_o = np.where(orders_==int(o))[0][0]
            #print(x_o)
            ord = orders[x_o]
            #print(ord)
            name =  onlyfiles_keys[i].split('_')[3]+onlyfiles_keys[i].split('_')[4] + ' ' + ord
            df_list.append(exp_data_all[i][['energy_list',ord]].rename(columns={ord: name}))
            names.append(name)
    #    fig.add_traces(px.line(exp_data_all[entry],x='energy_list',y=order,labels=onlyfiles_keys[entry]).data)
    df = reduce(lambda x, y: x.merge(y, on='energy_list'), df_list)
    #print(df)
    fig = px.line(df,x='energy_list',y=names)
    return fig

@callback(
    Output(component_id='order-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='crossfilter-energy--slider', component_property='value'),
)
def update_graph_(entry,energy):
    #df_list = []
    names = []
    dict_temp = {'orders':orders}
    for i in entry:
        name =  onlyfiles_keys[i].split('_')[3]+onlyfiles_keys[i].split('_')[4]
        df = exp_data_all[i].loc[exp_data_all[i]['energy_list'] == energy][orders]  
        #new_data = {name: df.iloc[i] for i in range(df.shape[0])}
        #print(df.values[0])
        #new_df = pd.DataFrame(new_data)
        #print(new_df)
        dict_temp[name] = df.values[0]
        #df_list.append(df.values[0])
        names.append(name)
    
    #print(pd.DataFrame(dict_temp))
    df = pd.DataFrame(dict_temp)
    fig = px.scatter(df,x='orders',y=names , title=str(energy)+str(' eV'),
                     )
    return fig

@callback(
    Output(component_id='fit_energy-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='fit_radio-buttons-order', component_property='value'),    
    Input(component_id='fit_width--slider', component_property='value'), 
    Input(component_id='fit_height--slider', component_property='value'),   
    Input(component_id='fit_E_Ni--slider', component_property='value'),   
    Input(component_id='fit_E_NiO--slider', component_property='value'),   
    Input(component_id='fit_N--slider', component_property='value'),      
)
def update_graph_fit(entry,order,width,height,E_Ni,E_NiO,N):    


    df_list = []
    names = []
    for i in entry:
        name =  onlyfiles_keys[i].split('_')[3]+onlyfiles_keys[i].split('_')[4]
        df_list.append(exp_data_all[i][['energy_list',order]].rename(columns={order: name}))
        names.append(name)
    #    fig.add_traces(px.line(exp_data_all[entry],x='energy_list',y=order,labels=onlyfiles_keys[entry]).data)

    real_Ni = (1 - f_n_Ni(real_energy+E_Ni))*1
    imag_Ni = f_k_Ni(real_energy+E_Ni)* 1
    real_NiO = (1 - f_n_NiO2(real_energy+E_NiO))*1
    imag_NiO = f_k_NiO2(real_energy+E_NiO)* 1
    
    SLD_core = sld_converter(1-real_Ni-imag_Ni*1j,real_energy) 
    SLD_shell = sld_converter(1-real_NiO-imag_NiO*1j,real_energy)
    fit_keys = {'energy_list':[],
                order:[]}
    q_max_temp = 2*np.pi/200*float(order)
    for i,energy in enumerate(real_energy):
        int_qmax = Intensity_q_max(q_max_temp,width,height,
                                            SLD_H2O[i], #keys['n_medium'], #keys['nk_H2O'][i], #keys['n_medium'],
                                            SLD_shell[i], #keys['n_shell'+str(energy).replace('.','p')],
                                            SLD_core[i], #keys['n_core'+str(energy).replace('.','p')], #keys['nk_Ni'][i], #keys['n_core'],
                                            N,
                                            200)
        fit_keys['energy_list'].append(energy)
        fit_keys[order].append(int_qmax*f_I0_Ni(energy)*1.4707847359645934e08)
    #print(pd.DataFrame(fit_keys))
    df_list.append(pd.DataFrame(fit_keys).rename(columns={order: 'calc'}))
    names.append('calc')

    df = reduce(lambda x, y: x.merge(y, on='energy_list'), df_list)
    #print(df)
    fig = px.line(df,x='energy_list',y=names)
    return fig

@callback(
    Output(component_id='fit_order-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='fit_crossfilter-energy--slider', component_property='value'),
    Input(component_id='fit_width--slider', component_property='value'), 
    Input(component_id='fit_height--slider', component_property='value'),   
    Input(component_id='fit_E_Ni--slider', component_property='value'),   
    Input(component_id='fit_E_NiO--slider', component_property='value'),   
    Input(component_id='fit_N--slider', component_property='value'),  
)
def update_graph_fit_(entry,energy,width,height,E_Ni,E_NiO,N):
    #df_list = []
    names = []
    dict_temp = {'orders':orders}
    for i in entry:
        name =  onlyfiles_keys[i].split('_')[3]+onlyfiles_keys[i].split('_')[4]
        df = exp_data_all[i].loc[exp_data_all[i]['energy_list'] == energy][orders]  
        #new_data = {name: df.iloc[i] for i in range(df.shape[0])}
        #print(df.values[0])
        #new_df = pd.DataFrame(new_data)
        #print(new_df)
        dict_temp[name] = df.values[0]
        #df_list.append(df.values[0])
        names.append(name)

    real_Ni = (1 - f_n_Ni(energy+E_Ni))*1
    imag_Ni = f_k_Ni(energy+E_Ni)* 1
    real_NiO = (1 - f_n_NiO2(energy+E_NiO))*1
    imag_NiO = f_k_NiO2(energy+E_NiO)* 1
    
    SLD_core = sld_converter(1-real_Ni-imag_Ni*1j,energy) 
    SLD_shell = sld_converter(1-real_NiO-imag_NiO*1j,energy)
    x_energy_exp = np.where(real_energy==energy)[0][0]
    int_qmax = Intensity_q_max(q_max,width,height,
                                            SLD_H2O[x_energy_exp], #keys['n_medium'], #keys['nk_H2O'][i], #keys['n_medium'],
                                            SLD_shell, #keys['n_shell'+str(energy).replace('.','p')],
                                            SLD_core, #keys['n_core'+str(energy).replace('.','p')], #keys['nk_Ni'][i], #keys['n_core'],
                                            N,
                                            200)
    dict_temp['calc'] = int_qmax*f_I0_Ni(energy)*1.4707847359645934e08
    names.append('calc')
    
    #print(pd.DataFrame(dict_temp))
    df = pd.DataFrame(dict_temp)
    fig = px.scatter(df,x='orders',y=names , title=str(energy)+str(' eV'),
                     )
    return fig


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
