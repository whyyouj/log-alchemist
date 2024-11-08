from dash import Dash, Input, Output,State, dcc, no_update, html
import dash_bootstrap_components as dbc
from system_log import event_freq, component_freq, user_freq, top_spikes, inactivity, potential_shut_down, correlation_analysis, error_analysis, top_spikes_analysis
class App:
    
    # Initialize the Dash app with Bootstrap theme
    def __init__(self):
        """
        Initializes the Dash application with Bootstrap styling and layout components.

        Function Description:
        Creates a new Dash application instance with Bootstrap theme and sets up the basic layout structure 
        including a sidebar and content area. Defines styling for both the sidebar and main content areas 
        using CSS properties.

        Input:
        - None (initializes with self reference)

        Output:
        - None (sets up instance attributes: self.app, self.sidebar, self.content)

        Note:
        - Instantiates core application components that are used by other methods
        - Creates the visual framework for the dashboard
        """
        
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define styles for the sidebar and content
        SIDEBAR_STYLE ={      
                "position": "fixed",
                "top": 0,
                "left": 0,
                "bottom": 0,
                "width": "10rem",
                "padding": "2rem 1rem",
                "background-color":"rgba(211, 211, 211, 0.5)",
        }
        
        CONTENT_STYLE = {
        "margin-left": "10rem",
        "margin-right": "2rem",
        "padding": "1rem 0.5rem",
        }
        
        # Create the sidebar with navigation links
        self.sidebar = html.Div(
            [
                html.H4("Logs"),
                html.Hr(),
            
                dbc.Nav(
                    [
                        dbc.NavLink("Mac", href="/", active = "exact"),
                        dbc.NavLink("Open Stack", href="/openstack", active = "exact")
                    ],
                    vertical = True,
                    pills = True,
                    ),
            ],
            
            style = SIDEBAR_STYLE,
           
        )
        # Create a content area that will be updated based on the URL
        self.content = html.Div(id="page-content", children=[], style=CONTENT_STYLE)
        
        # Set up the app layout and callbacks
        self.setup_app_layout()
        self.setup_callbacks()
        
    def setup_app_layout(self):
        """
        Establishes the main application layout structure.

        Function Description:
        Configures the primary layout of the Dash application by combining the URL location component,
        sidebar navigation, and main content area into a single container.

        Input:
        - None (uses self reference)

        Output:
        - None (sets self.app.layout)

        Note:
        - Must be called after sidebar and content components are defined
        - Changes to this function will affect the entire application structure
        """
        self.app.layout = html.Div([
            dcc.Location(id='url'),
            self.sidebar,
            self.content
        ])
    
    def setup_callbacks(self):
        """
        Configures the application's interactive behavior through callbacks.

        Function Description:
        Sets up the callback system that enables dynamic content updates based on user interaction.
        Currently implements URL routing to display different content based on the pathname.

        Input:
        - None (uses self reference)

        Output:
        - None (establishes callback functions)

        Note:
        - Callback functions are registered with the Dash app instance
        - Modifications here affect the application's interactive behavior
        """
        @self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname")]
        )
        
        def render_page_content(pathname):
            """
            Generates the appropriate content based on the current URL pathname.

            Function Description:
            Acts as a router that returns different page content depending on the URL pathname.
            Handles the main page ("/"), OpenStack page ("/openstack"), and 404 errors.

            Input:
            - pathname (str): The current URL pathname

            Output:
            - list: A list of Dash HTML components representing the page content

            Note:
            - Returns a 404 page if the pathname doesn't match any defined routes
            - Main page includes various analysis graphs and visualizations
            """
            if pathname == "/":
                return [
                    html.Div([
                        html.H1('Mac '),
                        html.Hr(),
                        html.H2('Frequency Analysis'),
                        dcc.Graph(figure = event_freq()),
                        dcc.Graph(figure = component_freq()),
                        dcc.Graph(figure = user_freq()),
                        html.H2('Anomaly Detection'),
                        dcc.Graph(figure = top_spikes()),
                        dcc.Graph(figure = top_spikes_analysis()),
                        dcc.Graph(figure = inactivity()),
                        dcc.Graph(figure = potential_shut_down()),
                        dcc.Graph(figure = error_analysis()),
                        html.H2('Correlation Analysis'),
                        dcc.Graph(figure = correlation_analysis())
                    ])
                ]
            elif pathname =="/openstack":
                return [
                    html.Div([
                        html.H1('Open Stack'),
                    ])
                ]
            else:
                return [
                    html.Div([
                        html.H1('404: page not found')
                    ])
                ]
    
    def run(self, debug = False):
        """
        Launches the Dash application server.

        Function Description:
        Starts the web server that hosts the Dash application, making it accessible via 
        a web browser at port 8080.

        Input:
        - debug (bool): Flag to enable/disable debug mode, defaults to False

        Output:
        - None (starts the server process)

        Note:
        - Server will continue running until manually stopped
        - Debug mode provides additional development features when enabled
        """
        self.app.run_server(debug = debug, port = 8080)
        


if __name__ == "__main__":
    app = App()
    app.run(debug = True)