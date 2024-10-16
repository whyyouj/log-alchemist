import spacy
import pandas as pd

# Load spaCy's small English Language Model

nlp = spacy.load("en_core_web_sm")

class QueryParser:
    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = [col.lower() for col in dataframe.columns]
    
    def parse_query(self, query):
        doc = nlp(query.lower())
        
        # Extract key components
        intent = self.identify_intent(doc)
        target_columns = self.extract_target_columns(doc)
        filters = self.extract_filters(doc)
        group_by = self.extract_group_by(doc)
        aggregation = self.extract_aggregation(doc, intent)
        sort = self.extract_sort(doc)
        limit = self.extract_limit(doc)
        
        # Construct the prompt template
        prompt_template = {
            "intent": intent,
            "target_columns": target_columns,
            "filters": filters,
            "group_by": group_by,
            "aggregation": aggregation,
            "sort": sort,
            "limit": limit
        }
        
        return prompt_template
    
    def identify_intent(self, doc):
        intents = {
            'describe': ['describe', 'summary', 'statistics', "summarise"],
            'count': ['how many', 'count', 'number of'],
            'average': ['average', 'mean', 'avg'],
            'max': ['maximum', 'max', 'highest', 'biggest', 'largest'],
            'min': ['minimum', 'min', 'lowest', 'smallest'],
            'filter': ['filter', 'show me', 'get', 'list'],
            'top_n': ['top', 'highest', 'largest'],
            'group_by': ['group by', 'per', 'by'],
            'sort': ['sort', 'order'],
        }
        query_text = doc.text.lower()
        for intent, keywords in intents.items():
            for phrase in keywords:
                if phrase in query_text:
                    return intent
        return 'unknown'
    
    def extract_target_columns(self, doc):
        columns = set()
        for token in doc:
            if token.text in self.columns:
                columns.add(token.text)
            elif token.lemma_ in self.columns:
                columns.add(token.lemma_)
        return list(columns) if columns else None
    
    def extract_filters(self, doc):
        """
        We assume that the user can only add 1 logical operator. Only one "AND" or "OR"
        """
        filters = []
        tokens = [token for token in doc]
        i = 0
        logical_operator = None
        while i < len(tokens):
            if tokens[i].text in self.columns:
                # New filter with a column name
                column = tokens[i].text
                i += 1

                # Skip any tokens until an operator
                while i < len(tokens) and tokens[i].lemma_ not in ["=", "==", 'be', 'equals', 'equal', 'is', 'not', 'greater', 'less', 'more', '>=', '<=', '>', '<']:
                    i += 1

                if i < len(tokens):
                    operator_token = tokens[i]
                    operator = self.map_operator(operator_token.lemma_)
                    i += 1
                else:
                    operator = '=='

                # Skip any tokens that are not NUM or NOUN
                while i < len(tokens) and tokens[i].pos_ not in ["NUM", "NOUN"]:
                    i += 1

                if i < len(tokens):
                    value_token = tokens[i]
                    if tokens[i].pos_ == "NUM":
                        value = self.parse_value(value_token.text.strip('\'"'))
                    else:
                        value = value_token.text
                    i += 1
                else:
                    value = None

                filters.append({
                    "column": column,
                    "operator": operator,
                    "value": value,
                    "logical_operator": logical_operator
                })
                logical_operator = None

            elif tokens[i].text.upper() in ['AND', 'OR']:
                # Handle logical operator
                logical_operator = tokens[i].text.upper()
                i += 1

                # Check what's next
                if i < len(tokens) and tokens[i].lemma_ in ["=", "==", 'be', 'equals', 'equal', 'is', 'not', 'greater', 'less', 'more', '>=', '<=', '>', '<']:
                        operator_token = tokens[i]
                        operator = self.map_operator(operator_token.lemma_)
                        i += 1

                # Skip tokens that are not NUM or NOUN
                while i < len(tokens) and tokens[i].pos_ not in ["NUM", "NOUN"]:
                    i += 1

                if i < len(tokens):
                    value_token = tokens[i]
                    if tokens[i].pos_ == "NUM":
                        value = self.parse_value(value_token.text.strip('\'"'))
                    else:
                        value = value_token.text
                    i += 1
                else:
                    value = None

                filters.append({
                    "column": column,
                    "operator": operator,
                    "value": value,
                    "logical_operator": logical_operator
                })
                logical_operator = None
            else:
                i += 1
        return filters if filters else None
    
    def map_operator(self, operator):
        operator_map = {
            'equal': '==',
            'equals': '==',
            'be': '==',
            'is': '==',
            'not': '!=',
            'greater': '>',
            'more': '>',
            'less': '<',
            'greater equal': '>=',
            'less equal': '<=',
            '>=': '>=',
            '<=': '<=',
            '>': '>',
            '<': '<'
        }
        return operator_map.get(operator, '==')
    
    def parse_value(self, value):
        try:
            # Try to parse as a number
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            # Return as string
            return value.strip('\'"')
    
    def extract_group_by(self, doc):
        group_by_words = ['per', 'by', 'group by']
        group_by_columns = []
        tokens = [token for token in doc]
        for i, token in enumerate(tokens):
            if token.text in group_by_words:
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if next_token.text in self.columns:
                        group_by_columns.append(next_token.text)
        return group_by_columns if group_by_columns else None
    
    def extract_aggregation(self, doc, intent):
        aggregations = {
            'average': ['average', 'mean', 'avg'],
            'max': ['maximum', 'max', 'highest', 'biggest', 'largest'],
            'min': ['minimum', 'min', 'lowest', 'smallest'],
            'count': ['count', 'number of', 'how many']
        }
        query_text = doc.text.lower()
        for function, keywords in aggregations.items():
            for word in keywords:
                if word in query_text:
                    # Try to find the column to aggregate
                    for token in doc:
                        if token.text in self.columns or token.lemma_ in self.columns:
                            return {
                                "function": function,
                                "column": token.text
                            }
                    # If no specific column is mentioned
                    return {
                        "function": function,
                        "column": None
                    }
        # If intent matches an aggregation but no keywords found
        if intent in aggregations:
            return {
                "function": intent,
                "column": None
            }
        return None
    
    def extract_sort(self, doc):
        sort_order = None
        sort_columns = []
        tokens = [token for token in doc]
        for i, token in enumerate(tokens):
            if token.lemma_ in ['sort', 'order', 'highest', 'largest', 'smallest', 'lowest', 'descending', 'ascending']:
                # Determine order
                if token.lemma_ in ['highest', 'largest', 'descending', 'descend']:
                    sort_order = 'descending'
                elif token.lemma_ in ['smallest', 'lowest', 'ascending', 'ascend']:
                    sort_order = 'ascending'
                # Look for columns to sort by
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    if next_token.text in self.columns:
                        sort_columns.append(next_token.text)
                continue
            # Also check for phrases like "top 5 by sales"
            if token.text == 'by' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.text in self.columns:
                    sort_columns.append(next_token.text)
        return {
            "columns": sort_columns if sort_columns else None,
            "order": sort_order
        } if sort_order or sort_columns else None
    
    def extract_limit(self, doc):
        limit = None
        offset = 0
        tokens = [token for token in doc]
        for i, token in enumerate(tokens):
            if token.text == 'top' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.like_num:
                    limit = int(next_token.text)
            elif token.text == 'limit' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.like_num:
                    limit = int(next_token.text)
            elif token.text == 'offset' and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.like_num:
                    offset = int(next_token.text)
        return {
            "count": limit,
            "offset": offset
        } if limit is not None or offset > 0 else None
    
    # def extract_output_format(self, doc):
    #     formats = ['table', 'list', 'summary', 'chart', 'graph']
    #     for token in doc:
    #         if token.text in formats:
    #             return token.text
    #     return None

if __name__ == "__main__":
    df = pd.read_csv('../../logs/Test/Mac_2k.log_structured.csv')
    parser = QueryParser(df)
    queries = [
        "How many rows are in the dataset?",
        "How many rows are there given that component is kernel or user?",
        "Filter component where it is equal to user",
        "Summarise the data in a summary."
    ]
    
    for q in queries:
        print(f"Query: {q}")
        print("Parsed Output:")
        print(parser.parse_query(q))
        print("-" * 80)


    