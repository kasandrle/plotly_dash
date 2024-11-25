import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objs as go
from os import listdir
from os.path import isfile, join, expanduser
from astropy.io import fits

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Generate an example CCD image
path = '/home/kas/Projects/JSR_grating/exp_data/20240512_JSR_dvlp/'
str_take_this_voltage = 'Cap'
onlyfiles_keys = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.fits') and f.startswith(str_take_this_voltage)]
i_ = 4#96+1#0#24#-50
file = onlyfiles_keys[i_]
path_file = join(path,file)
#cp12 = [np.array([640., 269.]), np.array([357., 266.])]
#lineprofile,energy,theta,cp_center = drf.find_lineprofile_along_order(path_file,width_line=40,plot=True,print_=True,darkimage=None,cp12=None)
#print(np.shape(lineprofile))
#open image file
img_data=fits.open(path_file,ext=2)
ccd_image = img_data[2].data  # Replace with your actual CCD image data

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("CCD Image Viewer"),
                width={"size": 6, "offset": 3},
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id="ccd-image"),
                width={"size": 10, "offset": 1},
            )
        ),
        dbc.Row(
            dbc.Col(
                html.Label("Select Line"),
                width={"size": 2, "offset": 5},
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Slider(
                    id="line-slider",
                    min=0,
                    max=ccd_image.shape[0] - 1,
                    step=1,
                    value=50,
                    marks={i: f"{i}" for i in range(0, ccd_image.shape[0], 10)},
                ),
                width={"size": 10, "offset": 1},
            )
        ),
    ],
    fluid=True,
)

# Define the callback to update the graph
@app.callback(
    Output("ccd-image", "figure"),
    Input("line-slider", "value")
)
def update_graph(selected_line):
    fig = go.Figure()

    # Add the CCD image as a heatmap
    fig.add_trace(
        go.Heatmap(
            z=ccd_image,
            colorscale='Viridis',
            colorbar=dict(title="Intensity"),
        )
    )

    # Highlight the selected line
    fig.add_trace(
        go.Scatter(
            x=np.arange(ccd_image.shape[1]),
            y=np.full(ccd_image.shape[1], selected_line),
            mode='lines',
            name='Selected Line',
            line=dict(color='red')
        )
    )

    fig.update_layout(
        title="CCD Image with Selected Line",
        xaxis_title="X-axis",
        yaxis_title="Y-axis",
    )

    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True,port=8052)

