# Useless #

# from transformers import pipeline
# import pandas as pd

# # Load Data
# log_df = pd.read_csv("./data/mac/Mac_2k.log_structured.csv").head(5)

# # Load a pre-trained summarization model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# def model_based_summary(log_message):
#     summary = summarizer(log_message, max_length=30, min_length=10, do_sample=False)[0]['summary_text']
#     return summary

# # Apply the pre-trained model to generate summaries for all log messages
# log_df['summary'] = log_df['Content'].apply(model_based_summary)
