from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.utilities import PythonREPL
from langchain_community.llms import Ollama
from regular_agent.agent_ai import Agent_Ai
from pandasai import SmartDataframe

import re

class Python_Ai:
    def __init__(self, model = "llama3.1", df =None, temperature=0):
        self.model = model
        self.temperature = temperature
        self.df = df
        
    def get_llm(self):
        return Agent_Ai(model=self.model, temperature=self.temperature, df=self.df)
        
    def pandas_agent(self,query, verbose = False, return_intermediate_steps=True, handle_parsing_error =True, max_iterations=1, allow_dangerous_code=True ):
        llm = self.get_llm().llm
        pd_agent = create_pandas_dataframe_agent(llm= llm, df = self.df, verbose=verbose, return_intermediate_steps=return_intermediate_steps, max_iterations=max_iterations, allow_dangerous_code= allow_dangerous_code, handle_parsing_error=handle_parsing_error)
        out = pd_agent.invoke(query)
        return out
    
    def pandas_ai_agent(self, query):
        llm = self.get_llm().llm

        # Running PandasAI
        pandas_ai = SmartDataframe(self.df, config = {
            "llm" : llm,
            "enable_cache" : False
        })

        out = pandas_ai.chat(query)
        return out
    
    
    def code_extractor(self, text):
        pattern =r"```[Pp]ython\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        return match
    
    def code_parser(self, python_code):
        try:
            python_code = python_code.group(1).split("\n")
            input_code = 'df=df \n'
        except:
            print("No code to input")
            return "No code available", ''
        
        for i in python_code:
            '''
            if plt.show() in i:
                plt.savefig('output.png)
                        or
                st.plotly_chart(plt, theme='streamlit', use_container_width=True)
            '''
    
            if "pd.read_csv(" in i:
                continue
            if "pd.DataFrame(" in i:
                continue
            if len(i) == 0:
                continue
            elif len(i) > 1 and "#" in i:
                i = i[0: i.find("#")]
                print(i)
                if len(i) >= 1:
                    input_code+=i+"\n"
            else:
                input_code += i+"\n"
        repl = PythonREPL(_globals={"df": self.df})
        print(input_code)
        out = repl.run(input_code)
        return out, input_code
    
    def code_executor_agent(self, query = "", prompt=""):
        llm = self.get_llm()
        if prompt == "":
            out = llm.query_agent(query=query)
        else:
            out = llm.prompt_agent(query=query, prompt=prompt)
        
        python_code = self.code_extractor(out[out.rfind("Final Answer"):])
        output, input_code = self.code_parser(python_code)
        return output, input_code
        

if __name__== "__main__":
    llm = Python_Ai()
    print(llm.get_llm())