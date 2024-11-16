# Vantage AI: The Log Alchemist

## What is Vantage AI?
Vantage AI is a multifunctional chatbot with a focus on log analysis!

Timely insights from system, audit, and transaction logs are essential for maintaining system health, troubleshooting issues, and ensuring security and compliance. Audit logs capture critical system access events, transaction logs record specific application activities, and system logs monitor general performance and errors. Analysing these logs manually is time-consuming and complex, particularly as log data volumes grow.

Vantage AI enables users to upload their own audit, transaction, and system logs, and quickly query, analyse, and interpret them through natural language interaction. Hence, Vantage AI can streamline troubleshooting, enhance audit reporting, and allow even non-technical users to investigate issues independently, automatically identifying patterns, providing insights, and suggesting solutions.

## How to use Vantage AI
- Upload your logs by selecting them from your files or inputting an absolute folder path.
- Using the provided dropdown at the top of the chat, select the log you wish to query and analyse.
- Query the log you selected! Vantage AI is able to answer questions on the selected log, provide summaries, analyse for anomalies, and even plot graphs for data visualisation!
- You may also input generic queries unrelated to your logs! Vantage AI will respond to them like a regular chatbot!

## Setting Up
- Clone this repository
- At the root of the repository `log-alchemist`, create a virtual environment
- Activate the virtual environment
- Install all the required dependencies in the virtual environment by running the command: `pip install -r requirements.txt`
- Pull the required model from Ollama: `ollama pull team25_vantage/multi_question_agent`; `ollama pull team25_vantage/pandasai_agent`; `ollama pull team25_vantage/final_agent`
- Navigate to the `app` directory
- Launch Vantage AI by running the command: `streamlit run app.py`
