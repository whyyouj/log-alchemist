from pandasai import Agent
import os, sys, ast
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from regular_agent.agent_ai import Agent_Ai
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.skills import skill
import tempfile
import pandas as pd
from langchain_core.tools import tool
from pandasai.llm import LLM
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from pandasai.prompts.base import BasePrompt
from pandasai.pipelines.pipeline_context import PipelineContext

class LangchainLLM(LLM):
    """
    Class to wrap Langchain LLMs and make PandasAI interoperable
    with LangChain.
    """

    def __init__(self, langchain_llm: BaseLanguageModel):
        self.langchain_llm = langchain_llm
        
    def code_formatter(self, code):
        MODEL3='jiayuan1/nous_llm'
        llm = Agent_Ai(model=MODEL3)
        query = """Your role is to extract the code portion and format it with:
```python

```
Ensure result = {"type": ... , "value": ...} includes only "type" values: "string", "number", "dataframe", or "plot"."""
        res = llm.query_agent(query= code + "\n" + query)
        return res
    def call(
        self, instruction: BasePrompt, context: PipelineContext = None, suffix: str = ""
    ) -> str:
        prompt = instruction.to_string() + suffix
        memory = context.memory if context else None
        prompt = self.prepend_system_prompt(prompt, memory)
        self.last_prompt = prompt
        prompt = prompt + """
        """
        
        res = self.langchain_llm.invoke(prompt)
        res = res.replace("</|im_end|>", "")
        res = res.replace("</s>", "")
        res = res.replace("</|im_start|>", "")
        if "```python" not in res:
            res = f"""```python 
            {res}
            ``` """
        # res = self.code_formatter(res)
        #print("[START_PROMPT]", prompt, '[END_PROMPT]')
        print('[OUT]', res, '[END_OUT]')
        return res #res.content if isinstance(self.langchain_llm, BaseChatModel) else res


    @property
    def type(self) -> str:
        return f"langchain_{self.langchain_llm._llm_type}"

@skill
def overall_summary(df):
    """
    Use this for any question regarding an Overall Summary
    The output type will be a string
    Args:
        df pd.DataFrame: A pandas dataframe 
    """
    import sweetviz as sv

    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        tempfile_path = f.name
        try:
            report = sv.analyze([df,'Logs'])
        except:
            try:
                report = sv.analyze(df)
            except Exception as e:
                raise(Exception)
        report.show_html(filepath=tempfile_path, layout='vertical', open_browser=False)
    result = {'type': 'Python_AI_Summary' , 'path': {tempfile_path}}

    return tempfile_path

@skill
def overall_anomaly(df):
    """
    Use this for any question regarding an Overall Anomaly
    The output type will be a string
    Args:
        df pd.DataFrame: A pandas dataframe 
    """
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
    ### 0. Number of Rows ###
    #########################
    anomalies['Number of rows'] = len(df)

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

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        tempfile_path = f.name
        plt.savefig(tempfile_path, bbox_inches='tight', dpi=300)
        
    return tempfile_path


class Python_Ai:
    def __init__(self, model, df=[], temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.df = df
        
    def get_llm(self):
        
        '''
        This function is to initialise the OLLAMA model
        '''
        
        return Agent_Ai(
            model=self.model, 
            temperature=self.temperature, 
            df=self.df
        )
    
    def pandas_legend(self):
        
        '''
        This function is to call the pandas ai agent. 
        The agent here does not have any skill
        '''
        
        llm  = self.get_llm().llm
        pandas_ai = Agent(
            self.df, 
            description = """
                You are a highly skilled data analysis agent, responsible for handling and answering various data-related queries. 
                For each query I provide, your task is to carefully analyze the data and return the most accurate and optimized solution.
                
                Your response should include:
                1. The Python code necessary to derive the answer from the data.
                
                Always take your time to think through the query before responding, and ensure the code is optimized for both readability and performance.
                
                Typical questions you will handle include requests like "How many rows are there in the dataset?" or "What are the top 5 events that occurred?" so ensure your answers are tailored to these types of queries.
            """,
            config={
                "llm":llm,
                "open_charts":False,
                "enable_cache" : False,
                "save_charts": True,
                "max_retries":5,
                "verbose": True,
                "response_parser": StreamlitResponse,
                "custom_whitelisted_dependencies": ["sweetviz", "numpy", "scipy", "pandas", "tabulate", "matplotlib", "datetime"]
            }
        )
        return pandas_ai
    
    def pandas_legend_with_skill(self):
        
        '''
        This function is to call the pandas ai agent.
        This agent has both summary and anomaly skill.
        '''
        
        llm  = LangchainLLM(self.get_llm().llm)

        pandas_ai = Agent(
            self.df, 
            description = """
            You are a data analyst that has been tasked with the goal of outputing python code.
            """,
            config={
                "llm":llm,
                "open_charts":False,
                "enable_cache" : False,
                "save_charts": True,
                "max_retries":5,
                "response_parser": StreamlitResponse,
                "custom_whitelisted_dependencies": ["sweetviz", "collections", "pytz"]
            }
        )
        pandas_ai.add_skills(overall_summary)
        pandas_ai.add_skills(overall_anomaly)
        return pandas_ai


if __name__=="__main__":
    import pandas as pd
    df = pd.read_csv("../../../EDA/data/mac/Mac_2k.log_structured.csv")
    ai = Python_Ai(df=df).pandas_ai_agent('how many users are there and who are the different users')
    print(ai[0].explain(), ai[1])