# Import packages
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

from os import listdir
from os.path import isfile, join
import numpy as np
from functools import reduce

# Incorporate data
path = './data/20241215'
#path = '/home/kas/Projects/Ni_project/exp_data_paper/20240524_YaoYang/'
onlyfiles_keys = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]
exp_data_all = []
E_all = []
order_all = []
theta_all = []

for file in onlyfiles_keys:
    file_path = join(path, file)
    df = pd.read_csv(file_path)#.groupby('file_list', as_index=False)
    file_uni = np.unique(df['file_list'])
    # Iterate over each unique 'file_list' value
    for f in file_uni:
        # Create a mask for the current 'file_list' value
        mask = df['file_list'] == f
        
        # Normalize 'order_list' for the current 'file_list' value by the value where 'order_list' is 1
        norm_value = df.loc[mask & (df['order_list'] == 1), 'intensity_list'].values[0]
        #print(norm_value)
        df.loc[mask, 'normalized_intensity_list'] = df.loc[mask, 'intensity_list'] / norm_value

    theta_all.append(np.unique(df['theta_list']))
    order_all.append(np.unique(df['order_list']))
    E_all.append(np.unique(df['energy_list']))
    exp_data_all.append(df)

orders = np.unique(np.concatenate(order_all))
energies = np.unique(np.concatenate(E_all))
thetas = np.unique(np.concatenate(theta_all))
#print('loaded')
# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('Photoresist exp data', className="text-primary text-center fs-3")
    ]),

    dbc.Row([
        dbc.Checklist(options=[{"label": x, "value": i} for i, x in enumerate(onlyfiles_keys)],
                      value=[0],
                      inline=True,
                      id='radio-buttons-final')
    ]),

    dbc.Row([
        html.Div('____________________________________________________', className="line")
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Checklist(options=[{"label": x, "value": x} for x in orders],
                          value=[orders[0]],
                          inline=False,
                          id='radio-buttons-order'),
            dbc.Checkbox( id='boolean-button', label='Normalize?', value=False ),
            dcc.Dropdown( id='scale-selector', options=[ {'label': 'Linear', 'value': 'linear'}, {'label': 'Log', 'value': 'log'} ], value='linear', clearable=False )
        ], width=2),

        dbc.Col([
            dcc.Graph(figure={}, id='theta-scan')
        ], width=10),

        dbc.Row([
            html.Div('____________________________________________________', className="line")
        ]),

        dbc.Col([
            dbc.Row([
                dbc.Checklist(options=[{"label": x, "value": x} for x in energies],
                              value=[energies[0]],
                              inline=True,
                              id='radio-buttons-energy')
            ]),
            dbc.Row([
                dbc.Checklist(options=[{"label": x, "value": x} for x in thetas],
                              value=[thetas[5]],
                              inline=True,
                              id='radio-buttons-theta')
            ]),
            dbc.Checkbox( id='boolean-button-1', label='Normalize?', value=False ),
            dcc.Dropdown( id='scale-selector-1', options=[ {'label': 'Linear', 'value': 'linear'}, {'label': 'Log', 'value': 'log'} ], value='linear', clearable=False )
        ], width=5),
        dbc.Row([
            dcc.Graph(figure={}, id='order-scan')
        ]),
    ]),
], fluid=True)

# Add controls to build the interaction
@callback(
    Output(component_id='theta-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='radio-buttons-order', component_property='value'),
    Input('scale-selector', 'value'),
    Input('boolean-button', 'value'),
)
def update_graph(entry, order,scale,norm):
    df_list = []
    names = []
    for i in entry:
        e_uni_temp = np.unique(exp_data_all[i]['energy_list'])
        for o in order:
            for e in e_uni_temp:
                mask = (exp_data_all[i]['order_list'] == o) & (exp_data_all[i]['energy_list'] == e)
                name = onlyfiles_keys[i] + ' ' + str(o)+ ' ' + str(e)
                if norm:
                    df_list.append(exp_data_all[i][['theta_list', 'normalized_intensity_list']][mask].rename(columns={'normalized_intensity_list': name}))
                else:
                    df_list.append(exp_data_all[i][['theta_list', 'intensity_list']][mask].rename(columns={'intensity_list': name}))
                names.append(name)
    
    df = reduce(lambda x, y: x.merge(y, on='theta_list', how='outer'), df_list)
    fig = px.scatter(df, x='theta_list', y=names)
    fig.update_xaxes(range=[np.min(thetas)-0.5, np.max(thetas)+0.5])
    fig.update_layout( yaxis_type=scale)
    return fig

@callback(
    Output(component_id='order-scan', component_property='figure'),
    Input(component_id='radio-buttons-final', component_property='value'),
    Input(component_id='radio-buttons-energy', component_property='value'),
    Input(component_id='radio-buttons-theta', component_property='value'),
    Input('scale-selector-1', 'value'),
    Input('boolean-button-1', 'value'),
)
def update_graph_(entry, energy, theta,scale,norm):
    df_list = []
    names = []
    for i in entry:
        for e in energy:
            for t in theta:
                mask = (exp_data_all[i]['theta_list'] == t) & (exp_data_all[i]['energy_list'] == e)
                file_list_temp_uni = np.unique(exp_data_all[i]['file_list'][mask])
                for f in file_list_temp_uni:
                    name = onlyfiles_keys[i] + ' ' + str(e) + ' ' + str(t) + ' ' + f
                    if norm:
                        df_list.append(exp_data_all[i][['order_list', 'normalized_intensity_list']][mask & (exp_data_all[i]['file_list'] == f)].rename(columns={'normalized_intensity_list': name}))
                    else:
                        df_list.append(exp_data_all[i][['order_list', 'intensity_list']][mask & (exp_data_all[i]['file_list'] == f)].rename(columns={'intensity_list': name}))
                    names.append(name)
    
    df = reduce(lambda x, y: x.merge(y, on='order_list', how='outer'), df_list)
    fig = px.scatter(df, x='order_list', y=names, title=f"{energy} eV, {theta}")
    fig.update_xaxes(range=[np.min(orders)-0.5, np.max(orders)+0.5])
    fig.update_layout(yaxis_type=scale)
    return fig

# Run the app
if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)
