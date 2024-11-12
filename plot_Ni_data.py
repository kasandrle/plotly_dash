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

# Incorporate data #
#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')
path = '/home/kas/Projects/Ni_project/exp_data_paper/20240524_YaoYang/'
onlyfiles_keys = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('p9.csv') and f.startswith('Ni_pattern_NiEdge_No3_')]
exp_data_all = []
for file in onlyfiles_keys:#[0::5]:
    #print(file)
    file = join(path, file)
    df = pd.read_csv(file)
    
    exp_data_all.append(df)
orders = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']

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
            dbc.RadioItems(options=[{"label": x, "value": x} for x in orders],
                       value=orders[0],
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

], fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='energy-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='radio-buttons-order', component_property='value'),    
)
def update_graph(entry,order):
    
    df_list = []
    names = []
    for i in entry:
        name =  onlyfiles_keys[i].split('_')[4]
        df_list.append(exp_data_all[i][['energy_list',order]].rename(columns={order: name}))
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
        name =  onlyfiles_keys[i].split('_')[4]
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


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
