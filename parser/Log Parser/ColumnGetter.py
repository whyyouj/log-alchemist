import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
import examples as eg


class ColumnGetter:
    sample_log = """
    Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4 
    Jun 14 15:16:02 combo sshd(pam_unix)[19937]: check pass; user unknown
    Jun 14 15:16:02 combo sshd(pam_unix)[19937]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4 
    Jun 15 02:04:59 combo sshd(pam_unix)[20882]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root
    Jun 15 02:04:59 combo sshd(pam_unix)[20884]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root
    Jun 15 02:04:59 combo sshd(pam_unix)[20883]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root
    Jun 15 02:04:59 combo sshd(pam_unix)[20885]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root
    Jun 15 02:04:59 combo sshd(pam_unix)[20886]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root
    Jun 15 02:04:59 combo sshd(pam_unix)[20892]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root
    Jun 15 02:04:59 combo sshd(pam_unix)[20893]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root
    """

    
    def _generate_prompt_template_default(self, log_row):
        log_query_template = PromptTemplate.from_template(
            """
            You are an expert in log analytics, and I need your help to prepare only the relevant column names when given a few rows of the same log data. I will use these columns in a LogParser. You simply have to read the rows of log data and tell me what are the possible column names. 

            Here are a few examples:
            log_data: Jul  1 09:00:55 calvisitor-10-105-160-95 kernel[0]: IOThunderboltSwitch<0>(0x0)::listenerCallback - Thunderbolt HPD packet for route = 0x0 port = 11 unplug = 0
            Response: ["Month", "Day", "Time", "User", "Component", "PID", "Content"]

            log_data: 2016-09-28 04:30:30, Info                  CBS    Loaded Servicing Stack v6.1.7601.23505 with Core: C:\Windows\winsxs\amd64_microsoft-windows-servicingstack_31bf3856ad364e35_6.1.7601.23505_none_681aa442f6fed7f0\cbscore.dll
            Response: ["Date", "Time", "Level", "Component", "Content"]

            Given the following rows of log data:
            {log}

            Context:
            - The log data may contain different formats depending on the source. 
            - The logs contain standard components like date, time, user, component/module, process ID, and log message (content).
            - Your task is to infer the columns based on the structure of the provided log examples.
            - The column **Content** should always refer to the actual log message or event description, and **must be the last column** in the format.

            Important:
            - Your only response is 1 list of column names in the form ['column1', 'column2', 'column3', ..., 'Content'] and no other text.
            - The last column **must be called "Content"** and must be the **last column** in the format.
            """
        )
        log_query_template.format(
            log = log_row, 
        )

        return log_query_template
    
    def _generate_prompt_template_fewshot(self, log_row):
        example_prompt = PromptTemplate(
            input_variables = ['log_data', 'response'],
            template = "Question: {log_data}\n{response}"
        )
        few_shot_prompt = FewShotPromptTemplate(
        examples = eg.examples,
        example_prompt = example_prompt,
        prefix = """"
                    You are an expert in log analytics, and I need your help to prepare a list of column names when given a few rows of log data. 
                    You simply have to read the rows of log data and tell me what are the column names in order. 

                    Context:
                    - Each new line in the log data is a row of data.
                    - You can refer to the example response provided for sample column names.
                    - Your task is to infer the columns based on the structure of the provided log examples.
                    - You should attempt to minimize the total number of columns.
                    - The column **Content** **must be the last column** in the format.

                    Here are a few examples:
                """,
        suffix = """
                    Given the following rows of log data:
                    {log}

                    Help me generate the list of column names.
                    
                    Important:
                    - You should **minimize** the number of columns.
                    - Column names generated must have length of less than **20 characters**.
                    - Only include alphabets in the column names.
                    - Your only response is the list of column names in the form ['column1', 'column2', ..., 'Content'] in order and no other text.
                    - The last column **must be called "Content"** and must be the **last column** in the format.
                """,
            input_variables = ["log"]
        )
        log_query_template = few_shot_prompt
        log_query_template.format(
            log = log_row, 
        )

        return log_query_template

    def get_column(self, model, log_data, prompt_method = 'default'):
        print(f"[INFO] Generating Columns, Model: {model}, Prompt Method: {prompt_method}")
        # Calling Llama 3.1 model to llm variable
        llm = OllamaLLM(model = model)

        if prompt_method == 'default':
            prompt = self._generate_prompt_template_default(log_data)
        elif prompt_method =='fewshot':
            prompt = self._generate_prompt_template_fewshot(log_data)

        chain = prompt | llm

        output = chain.invoke({"log":log_data})
        #print(output)
        return output
