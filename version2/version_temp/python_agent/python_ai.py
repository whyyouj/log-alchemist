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
from pandasai.responses import StreamlitResponse
from pandasai.pipelines.pipeline_context import PipelineContext
class LangchainLLM(LLM):
    """
    Class to wrap Langchain LLMs and make PandasAI interoperable
    with LangChain.
    """

    langchain_llm: BaseLanguageModel

    def __init__(self, langchain_llm: BaseLanguageModel):
        self.langchain_llm = langchain_llm

    def call(
        self, instruction: BasePrompt, context: PipelineContext = None, suffix: str = ""
    ) -> str:
        prompt = instruction.to_string() + suffix
        memory = context.memory if context else None
        prompt = self.prepend_system_prompt(prompt, memory)
        self.last_prompt = prompt
        
        res = self.langchain_llm.invoke(prompt)
        print("mm", prompt, 'mm.')
        print('m', res, '.')
        return res.content if isinstance(self.langchain_llm, BaseChatModel) else res

    @property
    def type(self) -> str:
        return f"langchain_{self.langchain_llm._llm_type}"



@skill
def summary_skill(df):
    """
    Use this for any question regarding an Overall Summary
    The output type will be a string
    Args:
        df pd.DataFrame: A pandas dataframe 
    """
    import sweetviz as sv
    # if type(df) != list:
    #     dfs = [df]
    # else:
    #     dfs = df
    # tempfile_paths = []
    # for i, d in enumerate(dfs):

    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        tempfile_path = f.name
        try:
            report = sv.analyze([df,'logs'])
        except:
            try:
                report = sv.analyze(df)
            except Exception as e:
                raise(Exception)
        report.show_html(filepath=tempfile_path, layout='vertical', scale=0.65, open_browser=False)
        # tempfile_paths.append(tempfile_path)
    result = {'type': 'Python_AI_Summary' , 'path': {tempfile_path}}

    return tempfile_path

class Python_Ai:
    def __init__(self, model = "codellama:7b", df=None, temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.df = df
        
    def get_llm(self):
        return Agent_Ai(
            model=self.model, 
            temperature=self.temperature, 
            df=self.df
        )
    
    def pandas_legend(self):
        llm  = self.get_llm().llm
        pandas_ai = Agent(
            self.df, 
            description = """
                You are a data analysis agent tasked with the main goal to answer any data related queries. 
                Everytime I ask you a question, you should provide the code to that specifically answers the question.
            """,
            config={
                "llm":llm,
                "open_charts":False,
                "enable_cache" : False,
                "save_charts": True,
                "max_retries":3,
                "response_parser": StreamlitResponse,
                "custom_whitelisted_dependencies": ["sweetviz"]
            }
        )
        #pandas_ai.add_skills(summarise_df)
        return pandas_ai
    
    def pandas_legend_with_summary_skill(self):
        llm  = LangchainLLM(self.get_llm().llm)

        pandas_ai = Agent(
            self.df, 

            config={
                "llm":llm,
                "open_charts":False,
                "enable_cache" : False,
                "save_charts": True,
                "max_retries":3,
                "response_parser": StreamlitResponse,
                "custom_whitelisted_dependencies": ["sweetviz"]
            }
        )
        pandas_ai.add_skills(summary_skill)
        return pandas_ai
    
    def dataframe_selector(self):
        llm  = self.get_llm().llm
        pandas_ai = Agent(
            self.df, 
            description = """
                You are a dataframe agent tasked with the main goal of returning the dataframe that I am referring to in my query. 
                Everytime I provide a query, you should select the dataframe that I am querying and return it.
            """,
            config={
                "llm":llm,
                "open_charts":False,
                "enable_cache" : False,
                "save_charts": True,
                "max_retries":3,
                "response_parser": StreamlitResponse,
                "custom_whitelisted_dependencies": ["sweetviz"]
            }
        )
        return pandas_ai
    
    def freq_tool(self, col_name):
        try:
            top_5_freq = self.df[col_name].value_counts().head(5)
            top_5_freq_df = pd.DataFrame(top_5_freq).reset_index()
            top_5_freq_df.columns = [col_name, 'Count']
            top_5_freq_df
            return top_5_freq
            
        except:
            return False    
        
    def anomaly_tool(self):
        try:
            self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
            time_series = self.df.set_index('Datetime').resample('min').size()
            time_series_df = pd.DataFrame(time_series.nlargest(5)).reset_index()
            time_series_df.columns = ["Datetime", "Count"]
            return time_series_df
        except:
            return False
        
    def anomaly_graph_tool(self):
        import plotly.express as px
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        time_series = self.df.set_index('Datetime').resample('min').size()

        fig = px.scatter(
            time_series, 
            x=time_series.index, 
            y=time_series, 
            labels={'x': 'Datetime', 'y': 'Frequency'},
            title='Overview of Usage'
        )
        
        fig.update_traces(
            mode='lines+markers',  # Add lines between the points
            marker=dict(size=3)    # Set the marker (dot) size smaller
        )
        
        dir = './exports/charts'
        os.makedirs(dir, exist_ok=True)
        image_name = 'anomaly_graph.png'
        image_path = os.path.join(dir, image_name)
        fig.write_image(image_path)
        return image_path
    
    def correlation_tool(self, list_variable):
        try:
            encode_data = pd.get_dummies(self.df[list_variable], drop_first=True)
            correlation_df = encode_data.corr()
            correlation_matrix_unstacked = correlation_df.unstack().sort_values(ascending=False)
            top_correlations = correlation_matrix_unstacked[correlation_matrix_unstacked!= 1].drop_duplicates()
            greater_than = top_correlations[top_correlations > 0.8]
            variables = []
            value = []
            for i in greater_than.items():
                temp_list = [i[0][0], i[0][1]]
                temp_list.sort()
                variables.append(', '.join(temp_list))
                value.append(f'{i[1]:.3g}')
                
            correlation_analysis_df = pd.DataFrame({"Variables":variables, "Coefficeint":value})
            return correlation_analysis_df
        except:
            return False 
    
    def get_summary_old(self):
        query=f"""This is the head of the dataframe: {self.df.head(10).to_json()}
                Based on the given data, analyze each key and determine which categorical variables are suitable for analysis. 
                - LineID is simply a row identifier and does not need to be included 
                - Exclude any columns related to date or time.
                - Exclude `Content` because both `EventId` and `EventTemplate` already serve as identifiers for `Content`.
                - Ensure that redundant or highly similar columns are not included (e.g., choose only either `EventId` or `EventTemplate`).
                The final answer should be a list of the selected categorical variables in the format:
                <Thought> ...
                <Answer> ["exact key name", "exact key name", ...]"""
        llm = self.get_llm()
        out = llm.query_agent(query= query)
        format_out = out[out.rfind('<Answer>')+ 9: out.rfind(']')+1]
        try:
            # Frequency Analysis
            out_list = ast.literal_eval(format_out)
            summary_dict = {'type' : "Python_AI_Summary",'Frequency Analysis':{}, 'Anomaly Analysis':{}, 'Correlation Analysis':{}}
            for i in out_list:
                freq_tool_out = self.freq_tool(i)
                if freq_tool_out is False:
                    continue
                else:
                    n = len(freq_tool_out)
                    key = f'Top {n} {i}:'
                    freq_analysis_dict = summary_dict.get("Frequency Analysis")
                    freq_analysis_dict[key] = freq_tool_out
            
            # Anomaly Analysis
            anomaly_analysis_dict = summary_dict.get("Anomaly Analysis")
            anomaly_out = self.anomaly_tool()
            if anomaly_out is not False:
                anomaly_analysis_dict[f"Top {len(anomaly_out)} Usage Rate"] = anomaly_out
            anomaly_graph_out = self.anomaly_graph_tool()
            if anomaly_graph_out is not False:
                anomaly_analysis_dict['GRAPH'] = anomaly_graph_out
            
            #Correlation Analysis
            correlation_analysis_dict = summary_dict.get('Correlation Analysis')
            correlation_out = self.correlation_tool(out_list)
            if correlation_out is not False:
                correlation_analysis_dict['Top Correlated Variables'] = correlation_out
                    
            return summary_dict
            
        except Exception as e:
            print(e)
            return '<Python Agent get Summary> Please Try Again'
        
    def get_summary(self):
        # from skimpy import skim
        # from pandas_profiling import ProfileReport
        #profile = ProfileReport(self.df, title="Pandas Profiling Report", explorative=True)
        import sweetviz as sv
        res = []
        for d in self.df:
            with tempfile.NamedTemporaryFile(suffix='.html', delete= False) as f:
                tempfile_path = f.name
                report = sv.analyze(d)
                report.show_html(filepath=tempfile_path, layout='vertical', scale=0.65, open_browser=False)
                res.append({'type': 'Python_AI_Summary' , 'path': tempfile_path}) 

        print('SWEETVIZ: ', res)


if __name__=="__main__":
    import pandas as pd
    df = pd.read_csv("../../../EDA/data/mac/Mac_2k.log_structured.csv")
    ai = Python_Ai(df=df).pandas_ai_agent('how many users are there and who are the different users')
    print(ai[0].explain(), ai[1])