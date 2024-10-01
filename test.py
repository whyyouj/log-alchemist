from regular_agent.agent_ai import Agent_Ai
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
    "What is the meaning of life?"
]

df = pd.read_csv("./data/Mac_2k.log_structured.csv")
questions2 =["hi"]

llm = Agent_Ai(model = 'llama3', df = df)
for i in questions2:
    print(i)
    print(llm.prompt_agent(i))
# print(llm.query_agent(prompt))