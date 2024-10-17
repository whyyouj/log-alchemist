import spacy
from spacy.matcher import Matcher

class QueryParser:
    def __init__(self):
        # Load spaCy English model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Define the intents and associated keywords
        self.intents = {
            'describe': ['describe', 'summary', 'statistics', 'summarise'],
            'count': ['how many', 'count', 'number of'],
            'average': ['average', 'mean', 'avg'],
            'max': ['maximum', 'max', 'biggest', 'largest'],
            'min': ['minimum', 'min', 'lowest', 'smallest'],
            'filter': ['filter', 'get', 'list', 'where', "given"],
            'top_n': ['top', 'highest', 'largest'],
            'group_by': ['group by', 'per', 'by'],
            'sort': ['sort', 'order'],
        }
        
        # Map intents to code templates
        self.intent_code_templates = {
            'describe': 'df.describe()',
            'count': 'df.shape[0]',
            'average': 'df["{column}"].mean()',
            'max': 'df["{column}"].max()',
            'min': 'df["{column}"].min()',
            'filter': 'df[df["{column}"] {operator} {value}]',
            'top_n': 'df.nlargest({n}, "{column}")',
            'group_by': 'df.groupby("{column}").agg({{ {aggregation} }})',
            'sort': 'df.sort_values(by="{column}", ascending={ascending})',
        }

    def identify_intent(self, query):
        query_lower = query.lower()
        matched_intents = set()
        for intent, keywords in self.intents.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matched_intents.add(intent)
        return matched_intents

    def extract_parameters(self, query, intents_detected):
        parameters = {}
        doc = self.nlp(query)
        
        # Extract numbers from the query
        numbers = [ent.text for ent in doc.ents if ent.label_ == 'CARDINAL']
        
        # Possible operators mapping
        operator_map = {
            'equal to': '==',
            'equals': '==',
            'equal': '==',
            'is': '==',
            'are': '==',
            'greater than': '>',
            'more than': '>',
            'less than': '<',
            'fewer than': '<',
            'greater than or equal to': '>=',
            'at least': '>=',
            'less than or equal to': '<=',
            'at most': '<=',
            'not equal to': '!=',
            'does not equal': '!=',
            'not': '!=',
        }
        
        # Default operator
        parameters['operator'] = '=='
        
        if 'average' in intents_detected or 'max' in intents_detected or 'min' in intents_detected:
            # Extract the last noun as the column name
            nouns = [token.text for token in doc if token.pos_ == 'NOUN']
            parameters['column'] = nouns[-1] if nouns else 'column_name'
        
        if 'filter' in intents_detected:
            for token in doc:
                if token.dep_ == 'dobj' and token.pos_ == 'NOUN':
                    parameters['column'] = token.text
                elif token.dep_ == 'prep' and token.head.dep_ == 'dobj':
                    parameters['operator'] = operator_map.get(token.text.lower(), '==')
                    value_token = token.nbor()
                    if value_token.ent_type_ == 'CARDINAL' or value_token.pos_ == 'NUM':
                        parameters['value'] = value_token.text
            parameters['column'] = parameters.get('column', 'column_name')
            parameters['value'] = parameters.get('value', 'value')
        
        if 'top_n' in intents_detected:
            # Extract 'n' from numbers
            parameters['n'] = numbers[0] if numbers else 'n'
            # Extract 'column' as the last noun in the query
            nouns = [token.text for token in doc if token.pos_ == 'NOUN']
            parameters['column'] = nouns[-1] if nouns else 'column_name'
        
        if 'sort' in intents_detected:
            # Extract 'column' after 'by' or 'on'
            for token in doc:
                if token.text.lower() in ['by', 'on']:
                    next_token = token.nbor()
                    if next_token.pos_ == 'NOUN':
                        parameters['column'] = next_token.text
                        break
            parameters['column'] = parameters.get('column', 'column_name')
            parameters['ascending'] = 'True' if 'ascending' in query.lower() or 'ascending order' in query.lower() else 'False'
        
        if 'group_by' in intents_detected:
            # Extract 'column' after 'by' or 'per'
            for token in doc:
                if token.text.lower() in ['by', 'per']:
                    next_token = token.nbor()
                    if next_token.pos_ == 'NOUN':
                        parameters['column'] = next_token.text
                        break
            parameters['column'] = parameters.get('column', 'column_name')
            # For aggregation, look for verbs like 'sum', 'average', 'mean', etc.
            aggregation_functions = ['sum', 'average', 'mean', 'count', 'min', 'max']
            aggregation = 'sum'  # default
            for token in doc:
                if token.lemma_ in aggregation_functions:
                    aggregation = token.lemma_
                    break
            parameters['aggregation'] = f"'column_to_aggregate': '{aggregation}'"
        
        return parameters

    def generate_code(self, intents_detected, parameters):
        # Prioritize 'top_n' over 'max' if both are detected
        if 'top_n' in intents_detected and 'max' in intents_detected:
            intents_detected.remove('max')

        code_snippets = []
        for intent in intents_detected:
            template = self.intent_code_templates.get(intent)
            if template:
                try:
                    code = template.format(**parameters)
                    code_snippets.append(code)
                except KeyError as e:
                    # Handle missing parameters
                    code_snippets.append(f"Missing parameter: {e}")
        return code_snippets

    def parse_query(self, query):
        intents_detected = self.identify_intent(query)
        if not intents_detected:
            prompt_template = f"""
            Given the user's query:
            "{query}"

            The query parser could not identify intent of query. 
            
            Please generate the final Python code that fulfills the user's request, taking your time to think about the solution. 

            Mention to the user to refine or clarify their query should they want a better respond.
            """
        else:
            parameters = self.extract_parameters(query, intents_detected)
            code_snippets = self.generate_code(intents_detected, parameters)
            
            # Prepare the prompt template
            code_snippets_formatted = '\n'.join(code_snippets)
            prompt_template = f"""
                Given the user's query:
                "{query}"

                The following code snippets are generated after parsing the query:
                {code_snippets_formatted}

                Please generate the final Python code that fulfills the user's request, using the code snippets as guidance.
            """
        return prompt_template

if __name__ == "__main__":
    parser = QueryParser()
    queries = [
        "How many rows are in the dataset?",
        "How many rows are there given that component is kernel or user?",
        "Filter component where it is equal to user",
        "Summarise the data in a summary.",
        "What is the sky colour?",
        "Show me the top 5 highest salaries",
        "What is the average age of employees?",
        "Filter the data where department is Sales",
        "Sort the data by salary in descending order",
        "Group the data by region and sum the sales",
        "List employees with age greater than 30",
    ]
    
    for q in queries:
        print(f"Query: {q}")
        print("Parsed Output:")
        print(parser.parse_query(q))
        print("-" * 80)


"""
The following code assumes that PandasAI can filter df and provide the df columns
"""
# Load spaCy's small English Language Model
# nlp = spacy.load("en_core_web_sm")

# class QueryParser:
#     def __init__(self, dataframe):
#         self.df = dataframe
#         self.columns = [col.lower() for col in dataframe.columns]
    
#     def parse_query(self, query):
#         doc = nlp(query.lower())
        
#         # Extract key components
#         intent = self.identify_intent(doc)
#         target_columns = self.extract_target_columns(doc)
#         filters = self.extract_filters(doc)
#         group_by = self.extract_group_by(doc)
#         aggregation = self.extract_aggregation(doc, intent)
#         sort = self.extract_sort(doc)
#         limit = self.extract_limit(doc)
        
#         # Construct the prompt template
#         prompt_template = {
#             "intent": intent,
#             "target_columns": target_columns,
#             "filters": filters,
#             "group_by": group_by,
#             "aggregation": aggregation,
#             "sort": sort,
#             "limit": limit
#         }
        
#         return prompt_template
    
#     def identify_intent(self, doc):
#         intents = {
#             'describe': ['describe', 'summary', 'statistics', "summarise"],
#             'count': ['how many', 'count', 'number of'],
#             'average': ['average', 'mean', 'avg'],
#             'max': ['maximum', 'max', 'highest', 'biggest', 'largest'],
#             'min': ['minimum', 'min', 'lowest', 'smallest'],
#             'filter': ['filter', 'show me', 'get', 'list'],
#             'top_n': ['top', 'highest', 'largest'],
#             'group_by': ['group by', 'per', 'by'],
#             'sort': ['sort', 'order'],
#         }
#         query_text = doc.text.lower()
#         for intent, keywords in intents.items():
#             for phrase in keywords:
#                 if phrase in query_text:
#                     return intent
#         return 'unknown'
    
#     def extract_target_columns(self, doc):
#         columns = set()
#         for token in doc:
#             if token.text in self.columns:
#                 columns.add(token.text)
#             elif token.lemma_ in self.columns:
#                 columns.add(token.lemma_)
#         return list(columns) if columns else None
    
#     def extract_filters(self, doc):
#         """
#         We assume that the user can only add 1 logical operator. Only one "AND" or "OR"
#         """
#         filters = []
#         tokens = [token for token in doc]
#         i = 0
#         logical_operator = None
#         while i < len(tokens):
#             if tokens[i].text in self.columns:
#                 # New filter with a column name
#                 column = tokens[i].text
#                 i += 1

#                 # Skip any tokens until an operator
#                 while i < len(tokens) and tokens[i].lemma_ not in ["=", "==", 'be', 'equals', 'equal', 'is', 'not', 'greater', 'less', 'more', '>=', '<=', '>', '<']:
#                     i += 1

#                 if i < len(tokens):
#                     operator_token = tokens[i]
#                     operator = self.map_operator(operator_token.lemma_)
#                     i += 1
#                 else:
#                     operator = '=='

#                 # Skip any tokens that are not NUM or NOUN
#                 while i < len(tokens) and tokens[i].pos_ not in ["NUM", "NOUN"]:
#                     i += 1

#                 if i < len(tokens):
#                     value_token = tokens[i]
#                     if tokens[i].pos_ == "NUM":
#                         value = self.parse_value(value_token.text.strip('\'"'))
#                     else:
#                         value = value_token.text
#                     i += 1
#                 else:
#                     value = None

#                 filters.append({
#                     "column": column,
#                     "operator": operator,
#                     "value": value,
#                     "logical_operator": logical_operator
#                 })
#                 logical_operator = None

#             elif tokens[i].text.upper() in ['AND', 'OR']:
#                 # Handle logical operator
#                 logical_operator = tokens[i].text.upper()
#                 i += 1

#                 # Check what's next
#                 if i < len(tokens) and tokens[i].lemma_ in ["=", "==", 'be', 'equals', 'equal', 'is', 'not', 'greater', 'less', 'more', '>=', '<=', '>', '<']:
#                         operator_token = tokens[i]
#                         operator = self.map_operator(operator_token.lemma_)
#                         i += 1

#                 # Skip tokens that are not NUM or NOUN
#                 while i < len(tokens) and tokens[i].pos_ not in ["NUM", "NOUN"]:
#                     i += 1

#                 if i < len(tokens):
#                     value_token = tokens[i]
#                     if tokens[i].pos_ == "NUM":
#                         value = self.parse_value(value_token.text.strip('\'"'))
#                     else:
#                         value = value_token.text
#                     i += 1
#                 else:
#                     value = None

#                 filters.append({
#                     "column": column,
#                     "operator": operator,
#                     "value": value,
#                     "logical_operator": logical_operator
#                 })
#                 logical_operator = None
#             else:
#                 i += 1
#         return filters if filters else None
    
#     def map_operator(self, operator):
#         operator_map = {
#             'equal': '==',
#             'equals': '==',
#             'be': '==',
#             'is': '==',
#             'not': '!=',
#             'greater': '>',
#             'more': '>',
#             'less': '<',
#             'greater equal': '>=',
#             'less equal': '<=',
#             '>=': '>=',
#             '<=': '<=',
#             '>': '>',
#             '<': '<'
#         }
#         return operator_map.get(operator, '==')
    
#     def parse_value(self, value):
#         try:
#             # Try to parse as a number
#             if '.' in value:
#                 return float(value)
#             else:
#                 return int(value)
#         except ValueError:
#             # Return as string
#             return value.strip('\'"')
    
#     def extract_group_by(self, doc):
#         group_by_words = ['per', 'by', 'group by']
#         group_by_columns = []
#         tokens = [token for token in doc]
#         for i, token in enumerate(tokens):
#             if token.text in group_by_words:
#                 if i + 1 < len(tokens):
#                     next_token = tokens[i + 1]
#                     if next_token.text in self.columns:
#                         group_by_columns.append(next_token.text)
#         return group_by_columns if group_by_columns else None
    
#     def extract_aggregation(self, doc, intent):
#         aggregations = {
#             'average': ['average', 'mean', 'avg'],
#             'max': ['maximum', 'max', 'highest', 'biggest', 'largest'],
#             'min': ['minimum', 'min', 'lowest', 'smallest'],
#             'count': ['count', 'number of', 'how many']
#         }
#         query_text = doc.text.lower()
#         for function, keywords in aggregations.items():
#             for word in keywords:
#                 if word in query_text:
#                     # Try to find the column to aggregate
#                     for token in doc:
#                         if token.text in self.columns or token.lemma_ in self.columns:
#                             return {
#                                 "function": function,
#                                 "column": token.text
#                             }
#                     # If no specific column is mentioned
#                     return {
#                         "function": function,
#                         "column": None
#                     }
#         # If intent matches an aggregation but no keywords found
#         if intent in aggregations:
#             return {
#                 "function": intent,
#                 "column": None
#             }
#         return None
    
#     def extract_sort(self, doc):
#         sort_order = None
#         sort_columns = []
#         tokens = [token for token in doc]
#         for i, token in enumerate(tokens):
#             if token.lemma_ in ['sort', 'order', 'highest', 'largest', 'smallest', 'lowest', 'descending', 'ascending']:
#                 # Determine order
#                 if token.lemma_ in ['highest', 'largest', 'descending', 'descend']:
#                     sort_order = 'descending'
#                 elif token.lemma_ in ['smallest', 'lowest', 'ascending', 'ascend']:
#                     sort_order = 'ascending'
#                 # Look for columns to sort by
#                 if i + 1 < len(tokens):
#                     next_token = tokens[i + 1]
#                     if next_token.text in self.columns:
#                         sort_columns.append(next_token.text)
#                 continue
#             # Also check for phrases like "top 5 by sales"
#             if token.text == 'by' and i + 1 < len(tokens):
#                 next_token = tokens[i + 1]
#                 if next_token.text in self.columns:
#                     sort_columns.append(next_token.text)
#         return {
#             "columns": sort_columns if sort_columns else None,
#             "order": sort_order
#         } if sort_order or sort_columns else None
    
#     def extract_limit(self, doc):
#         limit = None
#         offset = 0
#         tokens = [token for token in doc]
#         for i, token in enumerate(tokens):
#             if token.text == 'top' and i + 1 < len(tokens):
#                 next_token = tokens[i + 1]
#                 if next_token.like_num:
#                     limit = int(next_token.text)
#             elif token.text == 'limit' and i + 1 < len(tokens):
#                 next_token = tokens[i + 1]
#                 if next_token.like_num:
#                     limit = int(next_token.text)
#             elif token.text == 'offset' and i + 1 < len(tokens):
#                 next_token = tokens[i + 1]
#                 if next_token.like_num:
#                     offset = int(next_token.text)
#         return {
#             "count": limit,
#             "offset": offset
#         } if limit is not None or offset > 0 else None
    
#     def extract_output_format(self, doc):
#         formats = ['table', 'list', 'summary', 'chart', 'graph']
#         for token in doc:
#             if token.text in formats:
#                 return token.text
#         return None