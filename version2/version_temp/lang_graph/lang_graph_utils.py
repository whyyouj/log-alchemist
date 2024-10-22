import pandas as pd
import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai

graph_stage_prefix = '[STAGE]'

def router_agent(state: list):
    print(graph_stage_prefix, 'Router Agent')
    df = state['df']
    query = state['input']
    llm = Agent_Ai(model = 'llama3.1', df=df)
    out = llm.prompt_agent(query=query)
    print('ROUTER AGENT OUTPUT: ', out)
    return {"agent_out": out}

def router_agent_decision(state: list):
    router_out = state['agent_out']
    router_out = router_out.lower()
    out = router_out[router_out.rfind("answer") + 5:]
    if 'yes' in out.lower():
        return 'router_summary_agent'
    else:
        return 'final_agent'

def router_summary_agent(state: list):
    print(graph_stage_prefix, 'Router summary agent')
    llm = Agent_Ai(model='jiayuan1/summary_anomaly_llm_v3')
    query = state['input']
    # query_summary = f"""
    # You are suppose to determine if the <Question> is explicitly asking for a summary. When determining whether a question is asking for a summary, focus on whether the question is requesting a high-level overview of the data (summary), or if it’s asking for a specific value, action, or detail (non-summary). Always think before answering.
    
    # <Question> Is this asking for a summary: {query} 
    # <Thought> ...
    # <Answer> Always a Yes or No only
    # """
    out = llm.query_agent(query=query)
    out = out.lower()
    ans = out[out.rfind('answer')+ 5:]
    print('ROUTER SUMMARY AGENT OUTPUT: ', out)
    return {"agent_out": out}

def router_summary_agent_decision(state: list):
    router_out = state['agent_out']
    if 'summary' in router_out.lower():
        print('[INFO] Routed to summary agent')
        return 'python_summary_agent'
    elif 'anomaly' in router_out.lower():
        print('[INFO] Routed to anomaly agent')
        return 'python_anomaly_agent'
    else:
        print('[INFO] Routed to general agent')
        return 'python_pandas_ai'
    
def python_pandas_ai(state:list):
    print(graph_stage_prefix, 'Pandas AI agent')
    llm = state['pandas']
    query = state['input']
    prompt = f"""
    The following is the query from the user:
    {query}

    You are to respond with a code output that answers the user query. The code must not be a function and must not have a return statement.

    You are to following the instructions below strictly:
    - dfs: list[pd.DataFrame] is already provided.
    - Any query related to Date or Time, refer to the 'Datetime' column.
    - Any query related to ERROR, WARNING or EVENT, refer to the EventTemplate column.
    """
    out = llm.chat(prompt)
    return {"agent_out": out}

def python_summary_agent(state: list):
    print(graph_stage_prefix, 'Summary Agent')
    df = state['df']
    query = state['input']
    llm = Python_Ai(model = "llama3.1", df = df)
    pandasai_llm  = llm.pandas_legend_with_summary_skill()
    prompt = f"""
    The following is the query from the user:
    {query}

    If the query contains "summary", you must only execute the code for Sweetviz and output that result only.
    If the query does not contain "summary", you are to try your best to respond to the user query with an executable code.
    """
    out = pandasai_llm.chat(prompt) #state['pandas'].chat(prompt)
    print('PYTHON SUMMARY OUT: ', out)
    return {"agent_out": out}


def anomaly_skill(df):
    '''
    Use this for any question regarding an anomaly
    '''
    print('[INFO] Anomaly Skill called')
    import numpy as np
    from scipy import stats
    import pandas as pd
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    from datetime import datetime
    log_df = df
    anomalies = {}
    #########################
    ### 0. Number of rows ###
    #########################
    anomalies['Number of rows'] = len(log_df)
    
    ###########################################
    ### 1. Check for missing or null values ###
    ###########################################
    missing_values = pd.DataFrame(log_df.isnull().sum()).reset_index()
    missing_values.columns = ['Column', 'NA Count']
    missing_values = missing_values[missing_values['NA Count'] > 0]
    anomalies['missing_values'] = tabulate(missing_values, headers=missing_values.columns, tablefmt='pretty', showindex=False) if missing_values.shape[0] > 0 else 'No missing values'
    print('[INFO] Anomaly Skill: missing values checked')

    ################################
    ### 2. Detect duplicate rows ###
    ################################
    duplicate_rows = log_df[log_df.duplicated()]
    anomalies['duplicate_rows'] = len(duplicate_rows)
    print('[INFO] Anomaly Skill: duplicate rows checked')

    ##############################################
    ### 3. Identify and handle numeric columns ###
    ##############################################
    numeric_columns = log_df.select_dtypes(include=[np.number]).columns
    numeric_columns = numeric_columns[~numeric_columns.str.contains('id', case=False)]

    if not numeric_columns.empty:
        outliers_data = []

        for column in numeric_columns:
            correlation = np.corrcoef(log_df.index, log_df[column])[0, 1]
            
            # If the correlation is close to 1, skip outlier detection for this column
            if np.abs(correlation) > 0.95:  # Threshold to decide if it's linear
                continue

            Q1 = log_df[column].quantile(0.25)
            Q3 = log_df[column].quantile(0.75)
            
            IQR = Q3 - Q1
 
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_in_column = log_df[(log_df[column] < lower_bound) | (log_df[column] > upper_bound)]
           
            outliers_in_column['distance_from_bound'] = outliers_in_column[column].apply(
                lambda x: lower_bound - x if x < lower_bound else x - upper_bound
            )
            
            outliers_sorted = outliers_in_column.sort_values(by='distance_from_bound', ascending=False)
            
            # Select the top 3 most extreme outliers for the column
            top_3_outliers = outliers_sorted.head(3)
            
            # 11. Append the top 3 outliers for this column to the list (row number, column, value, distance)
            for index, row in top_3_outliers.iterrows():
                outliers_data.append({
                    'numeric_column': column,
                    'row_number': index,
                    'outlier_value': row[column]
                })

        if outliers_data:
            outliers_df = pd.DataFrame(outliers_data)
        else:
            outliers_df = pd.DataFrame(columns=['numeric_column', 'row_number', 'outlier_value'])
        
        anomalies['numeric_outliers'] = tabulate(outliers_df, headers=outliers_df.columns, tablefmt='pretty', showindex=False) if not outliers_df.empty else 'No numerical anomalies'
    print('[INFO] Anomaly Skill: numerical outliers checked')

    ##################################
    ### 4. Infer timestamp columns ###
    ##################################
    timestamp_columns = []
    for col in log_df.columns:
        try:
            if 'date' in col.lower() or 'time' in col.lower(): 
                timestamp_columns.append(col)
        except Exception:
            continue

    def timing_resampler(df, interval, column):
        interval_counts = df.resample(interval, on=column).size()
        interval_df = pd.DataFrame(interval_counts).reset_index()
        interval_df.columns = [f'Interval_{interval}', 'Count']
        skew = interval_df['Count'].skew()
        kurt = interval_df['Count'].kurt()
        return (interval_df, skew, kurt, skew+ np.abs(kurt-3))

    def timing_outliers(interval_df):
        q1 = interval_df['Count'].quantile(0.25)
        q3 = interval_df['Count'].quantile(0.75)
        iqr = q3-q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered_df = interval_df[(interval_df['Count'] > upper) | (interval_df['Count'] < lower)].reset_index(drop=True)
        return filtered_df

    def best_timing(df, ts_col):
        intervals = ['1min', '5min', '10min', '30min', '1H', '2H', '3H', '6H', '12H', '1D']
        res = []
        for i in intervals:
            data, skew, kurt, total = timing_resampler(df, i , ts_col)
            if len(data) == 0:
                pass
            res.append((i, data, skew, kurt, total))

        if len(res) == 0:
            return 'NA', pd.DataFrame()

        res.sort(key=lambda x:x[-1])
        best_interval = res[0][0]
        best_df = timing_outliers(res[0][1])
        return best_interval, best_df

    if timestamp_columns:
        for ts_col in timestamp_columns:
            log_df[ts_col] = pd.to_datetime(log_df[ts_col], errors='coerce')
            interval, data = best_timing(log_df, ts_col)
            if len(data) != 0 and interval != 'NA':
                data = data.sort_values(by='Count', ascending=False)
                anomalies[f'timestamp_freq_anomaly_{ts_col}_{interval}'] = tabulate(data, headers=data.columns, tablefmt='pretty', showindex=False)
    print('[INFO] Anomaly Skill: timestamp checked')

    ##################################################################
    ### 5. Identify categorical columns and detect rare categories ###
    ##################################################################
    categorical_columns = log_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns[:10]:
        event_frequency = log_df[col].value_counts()
        event_frequency_df = pd.DataFrame(event_frequency).reset_index()
        if len(event_frequency_df) <= 10:
            continue
        event_frequency_df.columns = ['Value', 'Count']
        event_frequency_df = event_frequency_df.sort_values('Count', ascending=True)
        rarest = event_frequency_df.head(3)
        anomalies[f'Rare_values_in_{col}'] = tabulate(rarest, headers=rarest.columns, tablefmt='pretty', showindex=False)
    print('[INFO] Anomaly Skill: categorical columns checked')

    #################################
    ### 6. Identify error columns ###
    #################################
    non_numeric_columns = log_df.select_dtypes(exclude=[np.number, np.datetime64]).columns
    error_results = []

    for column in non_numeric_columns:
        anomaly = log_df[log_df[column].str.contains('anomaly|anomalies', case=False, na=False)]
        anomaly_count = len(anomaly)
        error = log_df[log_df[column].str.contains('error|errors', case=False, na=False)]
        error_count = len(error)
        warning = log_df[log_df[column].str.contains('warning|warn', case=False, na=False)]
        warning_count = len(warning)
        res = [('anomalies',anomaly_count), ('errors', error_count), ('warnings', warning_count)]
        res_dic = {}
        for check, count in res:
            if count > 0:
                res_dic[check] = count
        if len(res_dic) > 0:
            error_results.append((column, str(res_dic)))

    if len(error_results) > 0:
        error_results_df = tabulate(error_results, headers=['Column', 'Error Counts'], tablefmt='grid')
        anomalies['Error Checks'] = error_results_df

    ########################
    ### add in timestamp ###
    ########################
    anomalies['ANOMALY CHECK DATE'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ##################################
    ### Ploting and returning path ###
    ##################################
    anomaly_table = [(key, str(value)) for key, value in anomalies.items()]
    anomaly_table = tabulate(anomaly_table, headers=['Check', 'Details'], tablefmt='grid')

    fig, ax = plt.subplots(figsize=(8, 4))  
    ax.axis('off')  
    plt.text(0.5, 0.5, anomaly_table, family='monospace', ha='center', va='center', fontsize=12)


    png_path = "tabulated_anomalies.png"
    plt.savefig(png_path, bbox_inches='tight', dpi=300)

    result = {'type': 'Python_AI_Anomaly' , 'path': {png_path}}
    return png_path


def python_anomaly_agent(state: list):
    print(graph_stage_prefix, 'Anomaly Agent')
    df = state['df']
    query = state['input']
    # llm = Python_Ai(model = "llama3.1", df = df)
    # pandasai_llm  = llm.pandas_legend_with_anomaly_skill() 
    # prompt = f"""
    # The following is the query from the user:
    # {query}

    # If the query contains "anomaly", you must only execute the code for anomaly and output that result only.
    # If the query does not contain "anomaly", you are to try your best to respond to the user query with an executable code.
    # """
    #prompt = 'Use your anomaly skill on the dataframe'
    #out = pandasai_llm.chat(query) 
    print(len(df))
    out = anomaly_skill(df[1])
    print('PYTHON ANOMALY OUT: ', out)
    return {"agent_out": out}

def router_python_output(state:list):
    router_out = state["agent_out"]
    if "Unfortunately, I was not able to answer your question, because of the following error:" in str(router_out):
        return "final_agent"
    else:
        return "__end__"
    
def final_agent(state:list):
    print(graph_stage_prefix, "Final Agent")
    llm = Agent_Ai(model = "llama3.1")
    query = state['input']
    prompt = f"""
    The following is the query from the user:
    {query}

    Try your best to answer the query. Take your time. If the query relates to any dataframe, assist accordingly to answer the query.
    """
    out = llm.query_agent(query=prompt)
    return {"agent_out":out}
    