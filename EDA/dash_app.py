# from dash import Dash, Input, Output,State, dcc, no_update, html
# import dash_bootstrap_components as dbc
# from system_log import event_freq, component_freq, user_freq, top_spikes, inactivity, potential_shut_down, correlation_analysis, error_analysis, top_spikes_analysis
# class App:
#     def __init__(self):
#         self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#         SIDEBAR_STYLE ={      
#                 "position": "fixed",
#                 "top": 0,
#                 "left": 0,
#                 "bottom": 0,
#                 "width": "10rem",
#                 "padding": "2rem 1rem",
#                 "background-color":"rgba(211, 211, 211, 0.5)",
#         }
#         CONTENT_STYLE = {
#         "margin-left": "10rem",
#         "margin-right": "2rem",
#         "padding": "1rem 0.5rem",
#         }
        
#         self.sidebar = html.Div(
#             [
#                 html.H4("Logs"),
#                 html.Hr(),
            
#                 dbc.Nav(
#                     [
#                         dbc.NavLink("Mac", href="/", active = "exact"),
#                         dbc.NavLink("Open Stack", href="/openstack", active = "exact")
#                     ],
#                     vertical = True,
#                     pills = True,
#                     ),
#             ],
            
#             style = SIDEBAR_STYLE,
           
#         )
        
#         self.content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)
#         self.setup_app_layout()
#         self.setup_callbacks()
        
#     def setup_app_layout(self):
#         self.app.layout = html.Div([
#             dcc.Location(id='url'),
#             self.sidebar,
#             self.content
#         ])
    
#     def setup_callbacks(self):
#         @self.app.callback(
#             Output("page-content", "children"),
#             [Input("url", "pathname")]
#         )
        
#         def render_page_content(pathname):
#             if pathname == "/":
#                 return [
#                     html.Div([
#                         html.H1('Mac '),
#                         html.Hr(),
#                         html.H2('Frequency Analysis'),
#                         dcc.Graph(figure = event_freq()),
#                         dcc.Graph(figure = component_freq()),
#                         dcc.Graph(figure = user_freq()),
#                         html.H2('Anomaly Detection'),
#                         dcc.Graph(figure = top_spikes()),
#                         dcc.Graph(figure = top_spikes_analysis()),
#                         dcc.Graph(figure = inactivity()),
#                         dcc.Graph(figure = potential_shut_down()),
#                         dcc.Graph(figure = error_analysis()),
#                         html.H2('Correlation Analysis'),
#                         dcc.Graph(figure = correlation_analysis())
#                     ])
#                 ]
#             elif pathname =="/openstack":
#                 return [
#                     html.Div([
#                         html.H1('Open Stack'),
#                     ])
#                 ]
#             else:
#                 return [
#                     html.Div([
#                         html.H1('404: page not found')
#                     ])
#                 ]
    
#     def run(self, debug = False):
#         self.app.run_server(debug = debug, port = 8080)
        


# if __name__ == "__main__":
#     app = App()
#     app.run(debug = True)

!pip install dash