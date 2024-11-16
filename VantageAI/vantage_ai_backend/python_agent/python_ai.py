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
    Wrapper class for making Langchain LLMs compatible with PandasAI.

    Function Description:
    Provides interface compatibility between Langchain language models and PandasAI
    by implementing required methods and handling code formatting.

    Input:
    - langchain_llm (BaseLanguageModel): Langchain language model instance

    Output:
    - None (creates wrapper instance)

    Note:
    - Fails silently if language model initialization fails
    - May return unformatted code if code extraction fails
    """
    def __init__(self, langchain_llm: BaseLanguageModel):
        """
        Initializes LangchainLLM wrapper instance.

        Function Description:
        Creates a wrapper instance that makes a Langchain language model compatible
        with PandasAI by providing necessary interface methods and code formatting
        capabilities.

        Input:
        - langchain_llm (BaseLanguageModel): Instance of a Langchain language model

        Output:
        - None (sets instance attribute)

        Note:
        - Fails silently if model instance is invalid
        - Required for PandasAI integration
        """
        self.langchain_llm = langchain_llm
        
    # def code_formatter(self, code):
    #     '''
    #     Description: Formats the code using a specific model.
        
    #     Input:
    #     - code: str
        
    #     Output:
    #     - res: formatted code as str
    #     '''
    #     MODEL3='jiayuan1/nous_llm'
    #     llm = Agent_Ai(model=MODEL3)
    #     query = """Your role is to extract the code portion and format it with:
    #         ```python

    #         ```
    #         Ensure result = {"type": ... , "value": ...} includes only "type" values: "string", "number", "dataframe", or "plot"."""
    #     res = llm.query_agent(query= code + "\n" + query)
    #     return res

    def code_formatter(self, code):
        """
        Formats code output from LLM responses.

        Function Description:
        Extracts and formats Python code from LLM responses, ensuring proper
        structure for PandasAI result dictionary formatting.

        Input:
        - code (str): Raw code output from LLM

        Output:
        - str: Formatted code block with proper Python markdown

        Note:
        - Returns original code if extraction pattern fails
        - Attempts to fix incomplete JSON structures
        """
        import re
        pattern = r"(import.*?result\s*=\s*\{.*?\})" #r"(import.*?result\s*=\s*\{.*?\})"

        # Search for the pattern in the text
        match = re.search(pattern, code, re.DOTALL)

        # Extract the matched code if found
        if match:
            extracted_code = match.group(1)
            
            pattern_2 = r"(result\s*=\s*\{.*?\})"
            match = re.search(pattern_2, extracted_code, re.DOTALL)
            if match:
                rs = match.group(1)
                count = 0 
                for i in rs:
                    if i =="{":
                        count += 1
                    elif i == "}":
                        count -=1
                if count != 0:
                    extracted_code+= "\"}"
            
            # print("[AI]", extracted_code)
            
            return f"```python \n {extracted_code}\n```"
            
        else:
            return code
    
    def call(
        self, instruction: BasePrompt, context: PipelineContext = None, suffix: str = ""
    ) -> str:
        """
        Executes LLM call with given instructions.

        Function Description:
        Processes instructions through the language model while handling proper
        formatting and cleaning of the response.

        Input:
        - instruction (BasePrompt): The prompt to send to the model
        - context (PipelineContext, optional): Execution context
        - suffix (str, optional): Additional prompt text

        Output:
        - str: Formatted response from the language model

        Note:
        - Returns empty string if model call fails
        - Removes special tokens from response
        """
        prompt = instruction.to_string() + suffix
        memory = context.memory if context else None
        prompt = self.prepend_system_prompt(prompt, memory)
        self.last_prompt = prompt
        prompt = prompt + """
        """
        prompt = prompt + """
        """
        
        res = self.langchain_llm.invoke(prompt)
        res = res.replace("</|im_end|>", "")
        res = res.replace("</s>", "")
        res = res.replace("</|im_start|>", "")
        # if "```python" not in res:
        #     res = f"""```python 
        #     {res}
        #     ``` """
        # res = self.code_formatter(res)
        #print("[START_PROMPT]", prompt, '[END_PROMPT]')
        # print('[OUT]', res, '[END_OUT]')
        res = self.code_formatter(res)
        return res #res.content if isinstance(self.langchain_llm, BaseChatModel) else res

    @property
    def type(self) -> str:
        """
        Returns the standardized type identifier for the Langchain LLM.

        Function Description:
        Generates a string identifier that combines the 'langchain' prefix with
        the underlying LLM type, providing a consistent way to identify the
        model type in PandasAI.

        Input:
        - None (uses instance attribute)

        Output:
        - str: Formatted string combining 'langchain_' prefix with model type

        Note:
        - Requires self.langchain_llm to be properly initialized
        - Returns malformed string if _llm_type not available
        """
        return f"langchain_{self.langchain_llm._llm_type}"

@skill
def overall_summary(df) -> str:
    """
    Generates comprehensive data summary using SweetViz.

    Function Description:
    Creates an interactive HTML report analyzing the DataFrame structure,s
    relationships, and statistics using the SweetViz library. The report
    includes correlations, distributions, and missing value analysis.

    Input:
    - df (pd.DataFrame): DataFrame to analyze

    Output:
    - tempfile_path (str): Path to generated HTML report file

    Note:
    - Raises Exception if report generation fails
    - Creates temporary file that needs manual cleanup
    - Report saved with 'vertical' layout for better readability
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
def overall_anomaly(df) -> str:
    """
    Performs comprehensive anomaly detection on DataFrame.

    Function Description:
    Analyzes DataFrame for various types of anomalies including:
    1. Missing values
    2. Duplicate rows
    3. Numerical outliers
    4. Timestamp patterns
    5. Rare categorical values
    6. Error patterns in text

    Input:
    - df (pd.DataFrame): DataFrame to analyze

    Output:
    - str: Path to generated PNG file containing anomaly report

    Note:
    - Saves visualization even if no anomalies detected
    """
    print('[INFO] Anomaly Skill called')
    import numpy as np
    from scipy import stats
    import pandas as pd
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    from datetime import datetime
    from copy import deepcopy
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
    numeric_columns = numeric_columns[~numeric_columns.str.contains('id|index', case=False)]

    if not numeric_columns.empty:
        outliers_data = []

        for column in numeric_columns:
            try:
                temp = log_df[column].dropna().sort_values()
                if len(temp) == 0:
                    continue
                comp = list(range(len(temp)))
                correlation = np.corrcoef(comp, temp)[0, 1]
                
                # If the correlation is close to 1, skip outlier detection for this column
                
                if np.abs(correlation) > 0.95 or np.isnan(correlation):  # Threshold to decide if it's linear
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
                

                for index, row in top_3_outliers.iterrows():
                    outliers_data.append({
                        'numeric_column': column,
                        'row_number': index,
                        'outlier_value': row[column]
                    })
            except Exception as err:
                print(err)

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
        """
        Resamples time series data at specified intervals.

        Function Description:
        Groups data by time intervals and calculates statistical measures (skewness, kurtosis)
        to identify unusual patterns in temporal distribution.

        Input:
        - df (pd.DataFrame): DataFrame containing timestamp data
        - interval (str): Time interval for resampling (e.g., '1min', '1H')
        - column (str): Name of timestamp column

        Output:
        - tuple: Contains:
            - interval_df: Resampled DataFrame
            - skew: Skewness measure
            - kurt: Kurtosis measure
            - total: Combined statistical measure

        Note:
        - Returns empty DataFrame if resampling fails
        - Requires datetime-formatted column
        """
        interval_counts = df.resample(interval, on=column).size()
        interval_df = pd.DataFrame(interval_counts).reset_index()
        interval_df.columns = [f'Interval_{interval}', 'Count']
        interval_df = interval_df.sort_values(by='Count')
        return interval_df

    def timing_outliers(interval_df):
        """
        Identifies temporal outliers using IQR method.

        Function Description:
        Calculates outlier bounds using interquartile range and filters
        data points that fall outside these bounds.

        Input:
        - interval_df (pd.DataFrame): DataFrame with count data by interval

        Output:
        - filtered_df (pd.DataFrame): DataFrame containing only outlier points

        Note:
        - Returns empty DataFrame if no outliers found
        - Uses 1.5 * IQR as outlier threshold
        """
        q1 = interval_df['Count'].quantile(0.25)
        q3 = interval_df['Count'].quantile(0.75)
        iqr = q3-q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered_df = interval_df[(interval_df['Count'] > upper) | (interval_df['Count'] < lower)].reset_index(drop=True)
        return filtered_df

    intervals = ['5min', '1H', '1D']
    if timestamp_columns:
        for ts_col in timestamp_columns:
            for i in intervals:
                try:
                    log_df[ts_col] = pd.to_datetime(log_df[ts_col], errors='coerce')
                    log_df_resampled = timing_resampler(log_df, i, ts_col)
                    outliers = timing_outliers(log_df_resampled)
                    if i == '1D':
                        print(log_df_resampled)
                    if len(outliers) != 0:
                        outliers = outliers.sort_values(by='Count', ascending=False)
                        outliers = outliers.head(3)
                        anomalies[f'timestamp_freq_anomaly_{ts_col}_{i}'] = tabulate(outliers, headers=outliers.columns, tablefmt='pretty', showindex=False)
                except Exception as err:
                    print(err)
                
    print('[INFO] Anomaly Skill: timestamp checked')

    ##################################################################
    ### 5. Identify categorical columns and detect rare categories ###
    ##################################################################
    categorical_columns = log_df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        try:
            event_frequency = log_df[col].value_counts()
            event_frequency_df = pd.DataFrame(event_frequency).reset_index()
            if len(event_frequency_df) <= 10:
                continue
            event_frequency_df.columns = ['Value', 'Count']
            event_frequency_df = event_frequency_df.sort_values('Count', ascending=True)
            rarest = event_frequency_df.head(3)
            rarest['Value'] = rarest['Value'].apply(lambda x: x[:200])
            anomalies[f'Rare_values_in_{col}'] = tabulate(rarest, headers=rarest.columns, tablefmt='pretty', showindex=False)
        except Exception as err:
            print(err)

    print('[INFO] Anomaly Skill: categorical columns checked')

    #################################
    ### 6. Identify error columns ###
    #################################
    non_numeric_columns = log_df.select_dtypes(include=['object']).columns
    error_results = []

    for column in non_numeric_columns:
        try:
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
        except Exception as err:
            print(err)

    if len(error_results) > 0:
        error_results_df = tabulate(error_results, headers=['Column', 'Error Counts'], tablefmt='grid')
        anomalies['Error Checks'] = error_results_df

    ##################
    ### Disclaimer ###
    ##################
    anomalies['DISCLAIMER'] = "1. For a detailed overview on the rare categories in each column, please query for a summary of the data and refer to the SweetVIZ summary. \n2. For rare categories, some values may be cut off if it is too long."

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
    """
    Data analysis wrapper class.

    Function Description:
    Provides interface for AI-driven data analysis using language models,
    incorporating both general pandas operations and specialized skills.

    Input:
    - model (str): Name/path of language model to use
    - df (list, optional): List of DataFrames to analyze
    - temperature (float, optional): Model temperature setting

    Output:
    - None (initializes instance)

    Note:
    - Requires valid model path
    - Temperature affects response randomness
    """
    def __init__(self, model, df=[], temperature=0.1):
        """
        Initializes Python_Ai instance.

        Function Description:
        Sets up core attributes needed for AI-powered data analysis,
        including model configuration and data storage.

        Input:
        - model (str): Name/path of language model
        - df (list, optional): List of DataFrames to analyze
        - temperature (float, optional): Model randomness parameter

        Output:
        - None (sets instance attributes)

        Note:
        - Empty df list if none provided
        - Default temperature of 0.1 for consistent outputs
        """
        self.model = model
        self.temperature = temperature
        self.df = df
        
    def get_llm(self):
        """
        Creates language model instance.

        Function Description:
        Initializes and configures an Ollama model instance with
        specified parameters for data analysis.

        Input:
        - None (uses instance attributes)

        Output:
        - Agent_Ai: Configured language model instance

        Note:
        - Returns None if model initialization fails
        - Uses instance temperature setting
        """

        
        return Agent_Ai(
            model=self.model, 
            temperature=self.temperature, 
            df=self.df
        )
    
    # def pandas_legend(self):
    #     '''
    #     Description: Calls the pandas AI agent without any skill.
        
    #     Input: None
        
    #     Output:
    #     - pandas_ai: Agent object
    #     '''
        
    #     llm  = LangchainLLM(self.get_llm().llm)

    #     pandas_ai = Agent(
    #         self.df, 
    #         description = """
    #             You are a highly skilled data analysis agent, responsible for handling and answering various data-related queries. 
    #             For each query I provide, your task is to carefully analyze the data and return the most accurate and optimized solution.
                
    #             Your response should include:
    #             1. The Python code necessary to derive the answer from the data.
                
    #             Always take your time to think through the query before responding, and ensure the code is optimized for both readability and performance.
                
    #             Typical questions you will handle include requests like "How many rows are there in the dataset?" or "What are the top 5 events that occurred?" so ensure your answers are tailored to these types of queries.
    #         """,
    #         config={
    #             "llm":llm,
    #             "open_charts":False,
    #             "enable_cache" : False,
    #             "save_charts": True,
    #             "max_retries":5,
    #             "verbose": True,
    #             "response_parser": StreamlitResponse,
    #             "custom_whitelisted_dependencies": ["sweetviz", "numpy", "scipy", "pandas", "tabulate", "matplotlib", "datetime"]
    #         }
    #     )
    #     return pandas_ai
    
    def pandas_legend_with_skill(self):
        """
        Creates enhanced pandas analysis agent.

        Function Description:
        Initializes a PandasAI agent with additional skills for
        summary statistics and anomaly detection.

        Input:
        - None (uses instance attributes)

        Output:
        - Agent: Configured PandasAI agent with added skills

        Note:
        - Returns None if agent creation fails
        - Includes summary and anomaly detection capabilities
        """
        llm  = LangchainLLM(self.get_llm().llm)

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
                "max_retries": 3,
                "response_parser": StreamlitResponse,
                "custom_whitelisted_dependencies": ["sweetviz", "collections", "pytz"]
            }
        )
        pandas_ai.add_skills(overall_summary)
        pandas_ai.add_skills(overall_anomaly)
        return pandas_ai
