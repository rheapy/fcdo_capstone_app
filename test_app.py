import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate

df = pd.DataFrame({'Category': np.repeat(['A', 'B', 'C'], 100),
                         'Item': np.arange(300),
                         'Value': np.random.rand(300)})

# Add an initial Rank column (default 1 to len(df))
df['Rank'] = range(1, len(df) + 1)
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Sortable, Scrollable Table with Dynamic Rank"),
    dcc.Graph(id='line-graph'),
    dash_table.DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in ['Rank'] + list(df.columns[:-1])],  # Put Rank first
        data=df.to_dict('records'),
        fixed_rows={'headers': True},
        sort_action='native',  # Enable sorting by clicking headers
        style_table={
            'height': '600px',  # Adjust height as needed
            'overflowY': 'auto'  # Vertical scrollbar if too many rows
        },
        style_cell={
            'minWidth': '100px', 'width': '150px', 'maxWidth': '200px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold'
        }
    )
])


@app.callback(
    Output('table', 'data'),
    Input('table', 'sort_by')
)
def update_ranking(sort_by):
    # Start from the original dataframe
    dff = df.copy()

    # If user has sorted the table, apply that sort
    if sort_by:
        for s in reversed(sort_by):  # Apply sorting in reverse order (multi-column sort)
            dff = dff.sort_values(
                by=s['column_id'],
                ascending=s['direction'] == 'asc',
                kind='mergesort'  # Stable sort
            )

    # Recompute the Rank based on current order
    dff['Rank'] = range(1, len(dff) + 1)

    # Reorder columns so Rank is first
    dff = dff[['Rank'] + [col for col in df.columns if col != 'Rank']]

    return dff.to_dict('records')


@app.callback(
    Output('line-graph', 'figure'),
    Input('table', 'data')
)
def update_line_graph(data):
    dff = pd.DataFrame(data)
    fig = px.line(dff, x='Item', y='Value')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)






#### test app for bootstrapping 

### UN voting process source code ### 
""" This scripts imports the results from the PCA and clustering of UN votes to 
make an app with plots, table and more information
It also enrich the data set with world bank data for population and GDP

"""
####################################
## 0. PACKAGES ##################
####################################

import pandas as pd
import numpy as np
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
from dash.dash_table import DataTable, FormatTemplate
import dash_bootstrap_components as dbc

from sklearn.metrics import pairwise_distances

####################################
## 1. IMPORT DATA ##################
####################################
pca_results = pd.read_csv('data_output/pca_results.csv')
pca_results = pca_results[pca_results['year'] >= 1990]

pca_results['cluster'] = pca_results['cluster'].astype('category') 

association_scores = pd.read_csv('data_output/association_scores.csv') 

mean_distance_table = association_scores.groupby('Country 1')['Mean distance'].mean().reset_index()
mean_distance_table = mean_distance_table.rename(columns={'Mean distance': 'Mean Mean Distance'})



#######################################################
## 2. UN PCA votes plot with year slider ##
#######################################################

#### Combined app testing 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

percentage = FormatTemplate.percentage(2)

app.layout = dbc.Container([
    
    html.H1('UN votes in the General Assembly', 
            style={'font-family': 'Helvetica', 'text-align': 'center'}),
    dbc.Row([
        # Left panel: filters and controls
        dbc.Col([
            html.H4('Nominal GDP per capita percentile', style={'font-family': 'Helvetica'}),
            dcc.RangeSlider(
                id='gdp-slider',
                min=0,
                max=1,
                value=[0, 1],
                marks={str(i): {'label': str(int(i * 100)) + '%', 
                                'style': {'font-family': 'Helvetica'}} 
                       for i in np.arange(0, 1.01, 0.05)},
            ),
            html.H4('Population percentile', style={'font-family': 'Helvetica', 'margin-top': '20px'}),
            dcc.RangeSlider(
                id='pop-slider',
                min=0,
                max=1,
                value=[0, 1],
                marks={str(i): {'label': str(int(i * 100)) + '%', 
                                'style': {'font-family': 'Helvetica'}} 
                       for i in np.arange(0, 1.01, 0.05)},
            ),
            html.H4('Select year', style={'font-family': 'Helvetica', 'margin-top': '20px'}),
            dcc.Slider(
                id='year-slider',
                min=pca_results['year'].min(),
                max=pca_results['year'].max(),
                value=pca_results['year'].max(),
                marks={str(i): {'label': str(i), 'style': {'font-family': 'Helvetica', 'writing-mode': 'vertical-lr'}} 
                       for i in range(pca_results['year'].min(), pca_results['year'].max(), 1)},
                step=1,
                included=False
            ),
            html.H4("Select Country", style={'font-family': 'Helvetica', 'margin-top': '20px'}),
            dcc.Dropdown(
                id='country-1-dropdown',
                value='United Kingdom',
                options=[{'label': c, 'value': c} for c in sorted(association_scores['Country 1'].unique())],
                clearable=False,
                style={'font-family': 'Helvetica'}
            ),
        ], md = 3, xs = 12,  style={'padding': '20px'}),  # Left panel width

        # Right panel: graph and tables
        dbc.Col([
            dcc.Graph(id='pca-graph', style={'font-family': 'Helvetica', 'height': '600px'}),

            html.Div(style={'height': '50px'}),  # Spacer

            html.H3("Country Pairwise Co-Cluster Table since 1990", style={'font-family': 'Helvetica'}),
            dash_table.DataTable(
                id='filtered-table',
                columns=[
                    {'name': 'Country 2', 'id': 'Country 2'},
                    dict(name='Co-Cluster Score', id='Co-Cluster Score', type='numeric', format=percentage),
                    dict(name='Mean distance', id='Mean distance', type='numeric', format=percentage)
                ],
                fixed_rows={'headers': True, 'data': 0},
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'font-family': 'Helvetica'},
                sort_action="native", 
                style_header={
                    'backgroundColor': 'lightgrey',
                    'fontWeight': 'bold'
                }
            ),

            html.Br(),  # Spacer

            html.H3("Mean Mean Distance Table", style={'font-family': 'Helvetica'}),
            dash_table.DataTable(
                id='mean-distance-table',
                columns=[
                    {'name': 'Country 1', 'id': 'Country 1'},
                    dict(name='Mean Mean Distance', id='Mean Mean Distance', type='numeric', format=percentage)
                ],
                fixed_rows={'headers': True, 'data': 0},
                data=mean_distance_table.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'font-family': 'Helvetica'},
                sort_action="native"
            )
        ], width=9, style={'padding': '20px'})  # Right panel width
    ])
], fluid=True)
@app.callback(
    Output('pca-graph', 'figure'),
    Input('gdp-slider', 'value'),
    Input('pop-slider', 'value'),
    Input('year-slider', 'value')
)
def update_figure(gdp_range, pop_range, year):
    pca_results_filtered = pca_results[(pca_results['gdp_pp'] >= gdp_range[0]) & (pca_results['gdp_pp'] <= gdp_range[1]) & (pca_results['pop'] >= pop_range[0]) & (pca_results['pop'] <= pop_range[1]) & (pca_results['year'] == year)]
    fig = px.scatter(
        pca_results_filtered,
        x='PCA1',
        y='PCA2',
        color='region_name',
        color_discrete_sequence=['rgb(17, 112, 170)', 'rgb(252, 125, 11)', 'rgb(163, 172, 185)', 'rgb(95, 162, 206)', 'rgb(200, 82, 0)', 'rgb(123, 132, 143)', 'rgb(163, 204, 233)', 'rgb(255, 188, 121)', 'rgb(200, 208, 217)'],
        hover_data={'ms_name': True, 
                    'region_name': True,
                     'PCA1': False, 'PCA2': False, 'pop': ':.2%', 'gdp_pp': ':.2%', 'cluster': False},
        labels={'PCA1': '', 'PCA2': ''},
        template='plotly_white'
    )
    fig.update_layout(
        font=dict(
            family='Helvetica',
            size=14,
            color='black'
        ),
        title_x=0.5,
        title={
            'text': f'UN countries relative position given their votes',
            'subtitle': {
                'text': f'Year: {year} <br> GDP PP range: {gdp_range[0]:.2%} to {gdp_range[1]:.2%} <br> Population range: {pop_range[0]:.2%} to {pop_range[1]:.2%} <br> Total countries: {len(pca_results_filtered)} <br> ',
            }
        }, 
        margin = dict(t = 200),
        xaxis=dict(
            showticklabels=False,
        ),
        yaxis=dict(
            showticklabels=False,
        ),
        legend_title_text = 'Continent',
    )
    return fig

@app.callback(
    Output('filtered-table', 'data'),
    Input('country-1-dropdown', 'value'),
)
def update_table(selected_country):
    filtered_df = association_scores[association_scores['Country 1'] == selected_country].copy()
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)

#######




# Calculate the correlation between Co-Cluster Score and Mean Distance
#correlation = association_scores[['Co-Cluster Score', 'Mean distance']].corr().iloc[0, 1]

# Print the total correlation
#print(f"Total Correlation between Co-Cluster Score and Mean Distance: {correlation:.2f}")


### Separate votes by country # most impurtant votes?
## Agregarle botón para seleccionar que países resaltar desde la lista completa para any givren year
# eso hay q hacerlo en das

# Mezclar todas las aplicaciones
