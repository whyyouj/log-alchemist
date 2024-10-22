from regular_agent.agent_ai import Agent_Ai
from python_agent.python_ai import Python_Ai
import pandas as pd
questions = [
    "How can I calculate the mean of the 'Price' column?",
    "Can I group the data by 'Category' and get the sum of 'Sales'?",
    "How do I filter rows where 'Age' is greater than 30?",
    "Is it possible to drop duplicate rows from the dataframe?",
    "Can you sort the dataframe by the 'Date' column in descending order?",
    "How do I merge two dataframes based on a common 'ID' column?",
    "Can I remove rows with missing values from the dataframe?",
    "How can I create a new column by multiplying two existing columns?",
    "What is the median value of the 'Revenue' column?",
    "Can you plot a bar graph of the 'Sales' column grouped by 'Region'?",
    "Hi, how are you?",
    "What’s the weather like today?",
    "Can you tell me a joke?",
    "Who won the game last night?",
    "What is the capital of France?",
    "Can you recommend a good book to read?",
    "What is your favorite color?",
    "How do I learn to play the guitar?",
    "Tell me about the history of the Internet.",
    "What is the meaning of life?",
    "hwo many outlier are there in the data set",
    'give me more insight in the data'
]
questions = ["explain why there are many peopple in the world"]
# df = pd.read_csv("../../data/Mac_2k.log_structured.csv")
# questions2 =["how many rows are there"]

llm = Agent_Ai(model = 'jiayuan1/summary_anomaly_llm_v2')
df = pd.read_csv('train2.csv')
# llm = Agent_Ai(model = 'jiayuan1/summary_anomaly_llm', df = df)
# llm= Python_Ai(model='mistral', df = df).pandas_legend_with_summary_skill()
correct = 0
total = 0
df = pd.read_csv('train2.csv')
for i in range(len(df)):
    # print(i)
    #print(llm.chat(i))
    print(i)
    out = llm.query_agent(f'What category does this question fall under: Summary, Anomaly, General:\n Question: {df.iloc[i].Input} \n Only output your Response')
    # out = llm.query_agent(df.iloc[i].Input)
    total += 1

    if out == df.iloc[i].Output:
        print(out)
        correct += 1
        
print(correct)
print(total)
#1733/1734

#out = llm.query_agent('What category does this question fall under: Summary, Anomaly, General: \nSummarise the data')
# from langchain_core.prompts import PromptTemplate
# prompt = PromptTemplate.from_template('What category does this question fall under: Summary, Anomaly, General:\n Question: summary of the data')

# out = llm.query_agent(query='what the people say')
# print(out)

# for i in questions:
#     # print(i)
#     #print(llm.chat(i))
    
#     out = llm.query_agent(f'What category does this question fall under: Summary, Anomaly, General: \n question: {i}')
#     print(i)
#     print(out)
