import spacy
import pandas as pd

# Load spaCy's small English Language Model
nlp = spacy.load("en_core_web_sm")

class QueryParser:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def parse_query(self, query):
        # Use spaCy to process the query
        doc = nlp(query.lower())

        # Convert the processed query back to a string for easier matching
        query_text = doc.text

        # Identify common intents based on keywords and structure
        if "how many" in query_text and ("where" in query_text or "given" in query_text):
            return self.parse_filter_count_query(doc)

        elif "how many" in query_text or "count" in query_text:
            return "len(df)"  

        elif "mean" in query_text and ("where" in query_text or "given" in query_text):
            return self.parse_filter_mean_query(doc)

        elif "mean" in query_text:
            column = self.extract_column_name(doc)
            if column:
                return f"df['{column}'].mean()"
            else:
                return None

        elif "filter" in query_text or "where" in query_text:
            return self.handle_filter(doc)

        elif "describe" in query_text or "summary" in query_text:
            return "df.describe()"

        else:
            return None

    def parse_filter_count_query(self, doc):
        """Parse queries like 'How many rows are there given that age > 30 AND sales < 300?'"""
        conditions = self.extract_conditions(doc)
        if conditions:
            return f"len(df[{conditions}])"
        else:
            return None

    def parse_filter_mean_query(self, doc):
        """Parse queries like 'What is the mean of sales where age > 30 AND sales < 400?'"""
        column = self.extract_column_name(doc)
        conditions = self.extract_conditions(doc)
        if column and conditions:
            return f"df[{conditions}]['{column}'].mean()"
        else:
            return None

    def extract_conditions(self, doc):
        """Extracts multiple conditions from the query (e.g., 'age > 30 AND sales < 300', 'component == kernel')."""
        conditions = []
        column = None
        operator = None
        value = None

        for token in doc:
            # Detect column name
            if token.pos_ == "NOUN":
                if column and operator and value:
                    # If a previous condition exists, append it
                    conditions.append(f"df['{column}'] {operator} {value}")
                if token.text.capitalize() in self.dataframe.columns:
                    column = token.text.capitalize()
                elif token.text in self.dataframe.columns:
                    column = token.text  # New condition starts here

            # Detect operator
            if token.text in [">", "greater than"]:
                operator = ">"
            if token.text in ["<", "less than"]:
                operator = "<"
            if token.text in ["=", "equals", "equal", "=="]:
                operator = "=="

            # Detect numerical value
            if token.text.isdigit():
                value = int(token.text)

            # Detect categorical value (either wrapped in quotes or identified as a categorical variable)
            if token.pos_ == "NOUN" and column:
                if column.capitalize() in self.dataframe:
                    column = column.capitalize()

                if (token.text in self.dataframe[column].unique() or
                    token.text.capitalize() in self.dataframe[column].unique()):
                    value = f"'{token.text}'"

            # Handle logical operators (AND, OR)
            if token.text in ["and", "or"]:
                if column and operator and value:
                    conditions.append(f"df['{column}'] {operator} {value}")
                    column = None
                    operator = None
                    value = None
                if token.text == "and":
                    conditions.append("&")
                if token.text == "or":
                    conditions.append("|")
        # Append the last condition
        if column and operator and value:
            conditions.append(f"df['{column}'] {operator} {value}")
        return " ".join(conditions)

    def extract_column_name(self, doc):
        """Extracts column names from the query using spaCy's entity recognition."""
        # Search for nouns in the query that match DataFrame column names
        for token in doc:
            if token.pos_ == "NOUN" and token.text in self.dataframe.columns:
                return token.text
        return None

    def handle_filter(self, doc):
        """Parse basic filter queries like 'Filter rows where age > 30'."""
        conditions = self.extract_conditions(doc)
        if conditions:
            return f"df[{conditions}]"
        else:
            return None

if __name__ == "__main__":
    df = pd.read_csv('../../logs/Test/Mac_2k.log_structured.csv')
    parser = QueryParser(df)
    queries = [
        "How many rows are in the dataset?",
        "How many rows are there given that component = kernel?",
        "Filter component where it is equal to kernel"
    ]
    
    for q in queries:
        print(parser.parse_query(q))
    