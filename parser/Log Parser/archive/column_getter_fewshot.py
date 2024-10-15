# ======== #
# Packages #
# ======== #
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
import examples as eg

'''
example_prompt = PromptTemplate(
    input_variables = ['log_data', 'response'],
    template = "Question: {log_data}\n{response}"
)
print(example_prompt.format(**eg.examples[0]))
few_shot_prompt = FewShotPromptTemplate(
    examples = eg.examples,
    example_prompt = example_prompt,
    prefix = """"
                You are an expert in log analytics, and I need your help to prepare only the relevant column names when given a few rows of the same log data. 
                I will use these columns in a LogParser. 
                You simply have to read the rows of log data and tell me what are the possible column names. 
                Here are a few examples:
            """,
    suffix = """
                Given the following rows of log data:
                {log}

                Context:
                - The log data may contain different formats depending on the source. 
                - Each new line in the log data is a row of data.
                - The logs contain standard components like date, time, user, component/module, process ID, and log message (content).
                - Your task is to infer the columns based on the structure of the provided log examples.
                - The column **Content** should always refer to the actual log message or event description, and **must be the last column** in the format.

                Important:
                - Your only response is the list of column names in the form ['column1', 'column2', 'column3', ..., 'Content'] and no other text.
                - The last column **must be called "Content"** and must be the **last column** in the format.
            """,
    input_variables = ["log"]
)
'''
# ========= #
# Functions #
# ========= #
def generate_prompt_template(log_row):
    example_prompt = PromptTemplate(
        input_variables = ['log_data', 'response'],
        template = "Question: {log_data}\n{response}"
    )
    few_shot_prompt = FewShotPromptTemplate(
    examples = eg.examples,
    example_prompt = example_prompt,
    prefix = """"
                You are an expert in log analytics, and I need your help to prepare only the relevant column names when given a few rows of the same log data. 
                I will use these columns in a LogParser. 
                You simply have to read the rows of log data and tell me what are the possible column names. 
                Here are a few examples:
            """,
    suffix = """
                Given the following rows of log data:
                {log}

                Context:
                - The log data may contain different formats depending on the source. 
                - Each new line in the log data is a row of data.
                - The logs contain standard components like date, time, user, component/module, process ID, and log message (content).
                - Your task is to infer the columns based on the structure of the provided log examples.
                - The column **Content** should always refer to the actual log message or event description, and **must be the last column** in the format.

                Important:
                - Your only response is the list of column names in the form ['column1', 'column2', 'column3', ..., 'Content'] and no other text.
                - The last column **must be called "Content"** and must be the **last column** in the format.
            """,
        input_variables = ["log"]
    )
    log_query_template = few_shot_prompt
    log_query_template.format(
        log = log_row, 
    )

    return log_query_template

def get_column(model, log_row):
    # Calling Llama 3.1 model to llm variable
    llm = OllamaLLM(model = model)

    prompt = generate_prompt_template(log_row)
    chain = prompt | llm

    print(chain.invoke({"log":log_row}))

# ======= #
# Testing #
# ======= #
log_row = """
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
