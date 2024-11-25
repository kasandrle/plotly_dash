# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
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
path = './data/'
all_results = pd.read_csv(join(path, 'ALL_RESULTS.csv'))
#print(all_results.columns)
onlyfiles_keys = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv') and (f.startswith('O') or f.startswith('1') or f.startswith('8'))]
exp_data_all = []
e_chem_state = []
e_chem_state_i = []
for i,file in enumerate(onlyfiles_keys):#[0::5]:
    #print(file)
    if file[:3] not in e_chem_state:
        e_chem_state.append(file[:3])
        e_chem_state_i.append(i)
    file = join(path, file)
    df = pd.read_csv(file)
    
    exp_data_all.append(df)
#print(e_chem_state)
#print(e_chem_state_i)
#print(exp_data_all[0].columns)
exp_col = []
fit_col = []
for entry in exp_data_all[0].columns:
    if entry.startswith('exp'):
        exp_col.append(entry)
    elif entry.startswith('fit'):
        fit_col.append(entry)
col_nk = ['real_Ni', 'imag_Ni', 'real_NiO','imag_NiO']
orders = ['2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0']

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('Ni fit results', className="text-primary text-center fs-3")
    ]),

    dbc.Row([
        dash_table.DataTable(
            data=all_results.to_dict('records'), 
            columns=[{"name": i, "id": i} for i in all_results.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            editable=True,
            selected_rows=[],
        )
    ]),

    dbc.Row([
        html.Div('____________________________________________________', className="line")
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Checklist(options=[{"label": x, "value": i} for i,x in enumerate(onlyfiles_keys)],
                        value=[0],
                        inline=False,
                        id='radio-buttons-final'),
            ], width=3),
        dbc.Col([           
            dcc.Graph(figure={}, id='nk-scan')
        ], width=8),

    ]),

    dbc.Row([
        html.Div('____________________________________________________', className="line")
    ]),




    dbc.Row([
        dbc.Col([
            html.Div("Orders:"),
            dbc.RadioItems(options=[{"label": x, "value": x} for x in orders],
                       value=orders[0],
                       inline=False,
                       id='radio-buttons-order')
        ], width=2),

        dbc.Col([
            dcc.Graph(figure={}, id='energy-scan')
        ], width=6),

       
    ]),
    dbc.Row([
        dbc.Col([
            html.Div("Energy slider:"),
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
    Output(component_id='nk-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'), 
)
def update_graph_nk(entry):    
    
    df_list_delta = []
    df_list_beta = []
    names_delta = []
    names_beta = []

    for i in entry:
        name =  onlyfiles_keys[i].split('.')[0]
        for j in col_nk:
            name_temp = name+'_'+j
            if j.startswith('r'):
                df_list_delta.append(exp_data_all[i][['energy_list',j]].rename(columns={j: name_temp}))
                names_delta.append(name_temp)
            elif j.startswith('i'):
                df_list_beta.append(exp_data_all[i][['energy_list',j]].rename(columns={j: name_temp}))
                names_beta.append(name_temp)
    #    fig.add_traces(px.line(exp_data_all[entry],x='energy_list',y=order,labels=onlyfiles_keys[entry]).data)
    df_delta = reduce(lambda x, y: x.merge(y, on='energy_list'), df_list_delta)
    df_beta = reduce(lambda x, y: x.merge(y, on='energy_list'), df_list_beta)
    #print(df)
    fig_delta = px.line(df_delta,x='energy_list',y=names_delta)
    fig_beta = px.line(df_beta,x='energy_list',y=names_beta)

    figures = [
                fig_delta,
                fig_beta
        ]

    fig = sp.make_subplots(rows=len(figures), cols=1) 

    for i, figure in enumerate(figures):
        for trace in range(len(figure["data"])):
            fig.append_trace(figure["data"][trace], row=i+1, col=1)
    return fig


@callback(
    Output(component_id='energy-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='radio-buttons-order', component_property='value'),    
)
def update_graph(entry,order):    
    e_chem_state = []
    e_chem_state_i = []
    for i in entry:
        file = onlyfiles_keys[i]
        if file[:3] not in e_chem_state:
            e_chem_state.append(file[:3])
            e_chem_state_i.append(i)
    #df_list = [exp_data_all[0][['energy_list','exp_'+order]].rename(columns={'exp_'+order: 'exp data'})]
    #names = ['exp data']
    df_list = []
    names = []
    for j,index in enumerate(e_chem_state_i):
        df_list.append(exp_data_all[index][['energy_list','exp_'+order]].rename(columns={'exp_'+order: 'exp data '+e_chem_state[j]}))
        names.append('exp data '+e_chem_state[j])
    for i in entry:
        name =  onlyfiles_keys[i].split('.')[0]
        df_list.append(exp_data_all[i][['energy_list','fit_'+order]].rename(columns={'fit_'+order: name}))
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
    e_chem_state = []
    e_chem_state_i = []
    for i in entry:
        file = onlyfiles_keys[i]
        if file[:3] not in e_chem_state:
            e_chem_state.append(file[:3])
            e_chem_state_i.append(i)

    #df_list = []
    names = []
    
    dict_temp = {'orders':orders,                 
                 }
    for j,index in enumerate(e_chem_state_i):
        df = exp_data_all[index].loc[exp_data_all[index]['energy_list'] == energy][exp_col]
        name = 'exp data '+e_chem_state[j]
        dict_temp[name] = df.values[0]
        names.append(name)
    for i in entry:
        name =  onlyfiles_keys[i].split('.')[0]
        df = exp_data_all[i].loc[exp_data_all[i]['energy_list'] == energy][fit_col]  
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
    app.run(host='0.0.0.0',debug=True,port=8051)
