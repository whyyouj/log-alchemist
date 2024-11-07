import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# File paths for the structured and template system logs
structured_system_log = '../logs/Mac/Mac_2k.log_structured.csv'
template_system_log = '../logs/Mac/Mac_2k.log_templates.csv'

# Read the relevant CSV files into DataFrames
structured_system_log_df = pd.read_csv(structured_system_log)
template_system_log_df = pd.read_csv(template_system_log)


# EventID
def event_freq():
    """
    Analyzes and visualizes the frequency of system events.

    Function Description:
    Calculates the top 5 most frequent events from the system logs and creates a combined 
    visualization with both a table showing event details and a bar chart showing event counts.
    The table includes Event IDs and their corresponding patterns from the template log.

    Input:
    - None (uses global structured_system_log_df and template_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A composite figure containing both table and bar chart
      of top 5 system events

    Note:
    - If no data is available, returns an empty figure
    - Requires both structured and template log DataFrames to be properly loaded
    """
    # Global DataFrame objects for storing log data
    event_freq = structured_system_log_df['EventId'].value_counts()
    top_5_event_freq = event_freq.head(5)
    top_5_event_freq_plotly = top_5_event_freq.reset_index()
    top_5_event_freq_plotly.columns = ['EventId', 'count']
    # Extract event explanations from template log
    explaination_list = []
    for i in top_5_event_freq_plotly['EventId']:
        explaination_list.append(template_system_log_df.loc[template_system_log_df['EventId'] == i]['EventTemplate'])
        
    fig = make_subplots(
    rows = 2,
    cols = 1,
    subplot_titles = ["","Event Count"],
    shared_xaxes = True,
    vertical_spacing = 0.003,
    specs=[[{"type":"table"}],
           [{"type":"bar"}]]
    )
    fig.add_trace(
        go.Table(
            columnwidth=[1, 6],
            
            header = dict(values=['Events ID', 'Event Patterns'], align = ['left']),
            cells = dict(values=[top_5_event_freq_plotly['EventId'],explaination_list],
                         font = dict(color=['rgb(45, 45, 45)'] * 5, size=8),
                         align = ['left'] * 5,)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=top_5_event_freq_plotly["EventId"],y=top_5_event_freq_plotly['count']), 
        row=2,
        col=1,
    )
    fig.update_layout(
        height = 500,
        showlegend=False,
        title_text = "Top 5 System Events"
    )
    return fig

def component_freq():
    """
    Analyzes and visualizes the frequency of system components.

    Function Description:
    Identifies and displays the top 5 most active components in the system logs through 
    a bar chart visualization, helping identify which system components are generating 
    the most log entries.

    Input:
    - None (uses global structured_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A bar chart showing the frequency of top 5 components

    Note:
    - Returns empty figure if no component data is available
    - Component names are displayed as-is from the log files
    """
    
    component_freq = structured_system_log_df['Component'].value_counts()
    top_5_component_freq = component_freq.head(5)
    top_5_component_freq_plotly = top_5_component_freq.reset_index()
    top_5_component_freq_plotly.columns = ['Component', 'count']

    fig = make_subplots(
    rows = 1,
    cols = 1,
    subplot_titles = ["Top 5 Logging Components"],
    )
    
    fig.add_trace(
        go.Bar(x=top_5_component_freq_plotly["Component"],y=top_5_component_freq_plotly['count']), 
    )
    return fig
    
def user_freq():
    """
    Analyzes and visualizes user activity frequency.

    Function Description:
    Creates a bar chart showing the top 5 most active users in the system logs,
    helping identify which users are generating the most system events.

    Input:
    - None (uses global structured_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A bar chart showing the frequency of top 5 users

    Note:
    - Returns empty figure if no user data is available
    - System processes may be included as users
    """
    
    user_freq = structured_system_log_df['User'].value_counts()
    top_5_user_freq = user_freq.head(5)
    top_5_user_freq_plotly = top_5_user_freq.reset_index()
    top_5_user_freq_plotly.columns = ['User', 'count']

    fig = make_subplots(
    rows = 1,
    cols = 1,
    subplot_titles = ["Top 5 Users"],
    )
    
    fig.add_trace(
        go.Bar(x=top_5_user_freq_plotly["User"],y=top_5_user_freq_plotly['count']), 
    )
    return fig

def top_spikes():
    """
    Identifies and visualizes periods of high system activity.

    Function Description:
    Analyzes the log data to find periods with unusually high numbers of events (spikes),
    creating a time series visualization with both a line chart showing overall activity
    and a table highlighting the top 5 spike periods.

    Input:
    - None (uses global structured_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A composite figure with line chart and table
      showing activity spikes

    Note:
    - Time series is resampled to minute intervals
    - Returns empty figure if datetime conversion fails
    """
    # Convert time strings to datetime objects for proper time series analysis
    structured_system_log_df['Datetime'] = pd.to_datetime(structured_system_log_df['Time'], format = '%H:%M:%S')
    time_series = structured_system_log_df.groupby(['Month', 'Date']).apply(
        lambda group: group.set_index('Datetime').resample('min').size()
    )

    # top_5_spikes = time_series.nlargest(5)
    # top_5_spikes

    time = time_series.reset_index(name='Count')
    time["DateTime"] = time['Date'].astype(str) + '-' + time['Month'] + " " + time['Datetime'].dt.time.astype(str)
    
    top_5_spikes = time.nlargest(5, columns = 'Count')
    # Create a line chart using plotly
    
    fig = make_subplots(
    rows = 2,
    cols = 1,
    shared_xaxes = True,
    vertical_spacing = 0.18,
    specs=[[{"type":"scatter"}],
           [{"type":"table"}]]
    )
    
    fig.add_trace(
        go.Scatter(x=time['DateTime'], y=time['Count'], mode='lines'),
        row = 1,
        col = 1,
    )
    
    fig.add_trace(
        go.Table(
            header = dict(
                values = ['Date', 'Frequency'],
                align='left',
            ),
            cells = dict(
                values= [list(top_5_spikes.DateTime),list(top_5_spikes.Count)],
                align='left',
            )
            
        ),
        row = 2,
        col = 1,
    )
    fig.update_layout(
        height = 500,
        showlegend=False,
        title_text = "Usage Spikes",
        xaxis=dict(tickfont = dict(size=6))
    )
    return fig

def top_spikes_analysis():
    """
    
    Description:
    This function identifies the events that occurred during the highest spike and generates a plotly table.
    
    Input: None
    
    Output:
    - fig: A plotly figure object containing the table.
    """
    fouth_of_july_df = structured_system_log_df[(structured_system_log_df['Month']=='Jul') & (structured_system_log_df['Date']== 4)]
    spike_df = fouth_of_july_df[(fouth_of_july_df["Time"] >= "23:22:00") & (fouth_of_july_df["Time"] < "23:23:00")]
    spike_df_plotly = spike_df.groupby(["EventId"])["EventId"].value_counts().reset_index()
    event_list = []
    for i in list(spike_df_plotly["EventId"]):
        temp_str = str(i)
        template_event =  template_system_log_df[template_system_log_df["EventId"] == i]["EventTemplate"]
        event_list.append(temp_str + ": \n" + template_event)
    fig = make_subplots(
        rows = 1,
        cols = 1,
        specs =[[{"type": "table"}]]
        )    
    fig.add_trace(go.Table(
    columnwidth=[3, 1],
    header = dict(values=['EventId and Type','Count'], align=['left']),
    cells = dict(values=[
        event_list, 
        list(spike_df_plotly['count'])
        ], align= ["left"], font = dict( size=10)
    )
    ),
                  row=1,
                  col=1)
    fig.update_layout(
        title_text = "Events that occured during the highest spike",
        height = 300
    )

    return fig


def inactivity():
    """
    Identifies and displays significant system inactivity periods.

    Function Description:
    Analyzes the log data to find the top 3 longest periods where no system events 
    were recorded, potentially indicating system downtime or issues.

    Input:
    - None (uses global structured_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A table showing the top 3 inactivity periods
      with their durations

    Note:
    - Requires datetime conversion of log entries
    - Returns empty figure if no significant gaps are found
    """
    date_time_structured_system_log_df = structured_system_log_df
    date_time_structured_system_log_df['DateTime'] = pd.to_datetime('2024 ' + date_time_structured_system_log_df['Month'] + " " + date_time_structured_system_log_df['Date'].astype(str) + " " + date_time_structured_system_log_df['Time'])
    time_diff = date_time_structured_system_log_df['DateTime'].diff().dropna()
    top_3_down_time = time_diff.nlargest(3).reset_index()
    period = []
    duration = []
    for i in top_3_down_time.values:
        start_date = f"{date_time_structured_system_log_df.iloc[i[0]-1]['Date']}-{date_time_structured_system_log_df.iloc[i[0]-1]['Month']}, {date_time_structured_system_log_df.iloc[i[0]-1]['Time']}"
        end_date = f"{date_time_structured_system_log_df.iloc[i[0]]['Date']}-{date_time_structured_system_log_df.iloc[i[0]]['Month']}, {date_time_structured_system_log_df.iloc[i[0]]['Time']}"
        period.append(f"{start_date} to {end_date}")
        duration.append(str(i[1]))
    
    fig = make_subplots(
    rows = 1,
    cols = 1,
    specs =[[{"type": "table"}]]
    )
    
    fig.add_trace(
        go.Table(
            columnwidth=[1, 1],
            
            header = dict(values=['Period', 'Duration'], align = ['left']),
            cells = dict(values=[period,duration],
                         font = dict(color=['rgb(45, 45, 45)'] * 5, size=10),
                         align = ['left'] * 5,)
        ),
        row = 1,
        col = 1,
    )
    fig.update_layout(
        height = 300,
        title_text = "Top 3 Periods of Inactivity"
    )
    return fig

def potential_shut_down():
    """
    Identifies potential system shutdown events.

    Function Description:
    Searches the log data for events containing shutdown-related keywords
    (shutdown, halt, poweroff, reboot) to identify system shutdown events.

    Input:
    - None (uses global structured_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A table showing details of potential
      shutdown events

    Note:
    - Keywords are case-insensitive
    - Returns empty figure if no shutdown events are found
    """
    shutdown_keywords = ['shutdown', 'halt', 'poweroff', 'reboot']
    shutdown_logs = structured_system_log_df[structured_system_log_df['Content'].str.contains('|'.join(shutdown_keywords), case=False, na=False)]
    fig = make_subplots(
        rows = 1,
        cols = 1,
        specs =[[{"type": "table"}]]
        )
    fig.add_trace(go.Table(
        columnwidth=[1, 2,1,3],
        header = dict(values=['DateTime','User', 'Component',  'Content'], align=['left']),
        cells = dict(values=[
            [f"{shutdown_logs['Date'].iloc[0]}-{shutdown_logs['Month'].iloc[0]}, {shutdown_logs['Time'].iloc[0]}"],
            [shutdown_logs['User']], 
            [shutdown_logs['Component']], 
            [shutdown_logs['Content']]
            ], align= ["left"], font = dict( size=10)
        )
        ))
    fig.update_layout(
        title_text = "Potential Shutdown",
        height = 300
    )

    return fig

def error_analysis():
    """
    Analyzes and visualizes system error patterns.

    Function Description:
    Identifies events containing error-related keywords and generates a frequency
    analysis of the top 5 most common error events.

    Input:
    - None (uses global structured_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A table showing the top 5 error events
      and their frequencies

    Note:
    - Only considers explicit 'error' keyword matches
    - Returns empty figure if no error events are found
    """
    error_keywords = ['error']
    error_logs = structured_system_log_df[structured_system_log_df['Content'].str.contains('|'.join(error_keywords), case=False, na=False)]
    error_logs_plotly = error_logs.groupby("EventId")["EventId"].value_counts().nlargest(5)
    error_logs_plotly = error_logs_plotly.reset_index()

    
    fig = make_subplots(
        rows = 1,
        cols = 1,
        specs =[[{"type": "table"}]]
        )    
    fig.add_trace(go.Table(
    columnwidth=[1, 1],
    header = dict(values=['EventId','Count'], align=['left']),
    cells = dict(values=[
        list(error_logs_plotly['EventId']), 
        list(error_logs_plotly['count'])
        ], align= ["left"], font = dict( size=10)
    )
    ),
                  row=1,
                  col=1)
    fig.update_layout(
        title_text = "Top 5 Events with potential error",
        height = 300
    )
    return fig


def correlation_analysis():
    """
    Analyzes relationships between system components, users, and events.

    Function Description:
    Performs correlation analysis on encoded categorical variables to identify
    potential relationships between different system elements. Creates a visualization
    combining a correlation heatmap and a table of top correlations.

    Input:
    - None (uses global structured_system_log_df)

    Output:
    - fig (plotly.graph_objects.Figure): A composite figure with correlation heatmap
      and top correlations table

    Note:
    - Uses one-hot encoding for categorical variables
    - May be computationally intensive for large datasets
    """
    
    # Convert categorical variables to numeric using one-hot encoding
    encode_data = pd.get_dummies(structured_system_log_df[['Component', 'User', 'EventId']], drop_first=True)
    correlation_df = encode_data.corr()
    correlation_matrix_unstacked = correlation_df.unstack().sort_values(ascending=False)
    top_correlations = correlation_matrix_unstacked[correlation_matrix_unstacked != 1].drop_duplicates()

    variables = []
    value = []
    for i in top_correlations.head(10).items():
        temp_list = [i[0][0], i[0][1]]
        temp_list.sort()
        variables.append(', '.join(temp_list))
        value.append(f'{i[1]:.3g}')
        
    fig = make_subplots(
        rows = 1,
        cols = 2,
        shared_xaxes = True,
        column_width= [0.5,0.5],
        specs=[[{"type":"table"}, {"type":"heatmap"}],
            ],
        subplot_titles= ["", "Heat Map"],
        )
    
    correlation_np = correlation_df.to_numpy
    fig.add_trace(go.Heatmap(
        z = correlation_df.values,
        colorscale = "Blues",
        colorbar=dict(
            len=1,        # Reduce the length of the colorbar
            thickness=10,   # Reduce the thickness of the colorbar
            y=0.5,          # Center the colorbar vertically
            yanchor='middle'
        ),
        showscale = True
        ),
            row= 1,
            col=2,
            )
    
    fig.add_trace(go.Table(
        header = dict(values=['Variables', 'Correlation Coefficient'], align= ['left']),
        cells = dict(values = [variables, value], align = ['left'], font = dict(size=10), height=30),
        
        ),
            row = 1,
            col = 1,
            )

    fig.update_xaxes(visible=False, constrain="domain", scaleanchor="y", row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)#domain=[0.5, 1],
    return fig





