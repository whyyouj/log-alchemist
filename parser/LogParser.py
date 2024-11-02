from ColumnGetter import ColumnGetter
from logparser import Drain
import re

class MaxRetriesError(Exception):
    pass

class LogParser:
    def __init__(self, model, prompt_method):
        """
        Description:
        This function initializes the LogParser class with the given model and prompt method.
        
        Input:
        - model: The model to be used for generating column names (str)
        - prompt_method: The prompt method to be used ('default' or 'fewshot') (str)
        
        Output: None
        """
        self.model = model
        self.prompt_method = prompt_method

    def read_file(self, path):
        """
        Description:
        This function reads the log file from the given path and returns its content as a string.
        
        Input:
        - path: The path to the log file (str)
        
        Output:
        - log: The content of the log file (str)
        """
        log = ''
        with open(path, mode='r') as f:
            for line in f:
                log += (line + '\n')
        return log
    
    def get_columns(self, path, max_retries=5, max_length=20, max_cols = 10):
        """
        
        Description:
        This function generates column names for the log data using the specified model and prompt method. It retries up to max_retries times if the generated columns do not meet the quality checks.
        
        Input:
        - path: The path to the log file (str)
        - max_retries: The maximum number of retries for generating column names (int, default=5)
        - max_length: The maximum length of a column name (int, default=20)
        - max_cols: The maximum number of columns (int, default=10)
        
        Output:
        - formatted_columns: The formatted column names (str)
        
        Raises:
        - MaxRetriesError: If no valid column names are generated after max_retries attempts
        """
        column_getter = ColumnGetter()
        log_str = self.read_file(path)

        for attempt in range(1, max_retries+1):
            print(f"[INFO] Attempt {attempt}: Generating Column Names...")
            cols = column_getter.get_column(self.model, log_str, self.prompt_method)

            match = re.search(r"\[(.*?)\]", cols)

            if match:
                columns_str = match.group(1)
                columns = [col.strip().strip('"').strip("'") for col in columns_str.split(',')]
                columns = [re.sub(r'[^A-Za-z]', '_', col) for col in columns]

                ### some quality checks
                if any(len(col) > max_length for col in columns):
                    print(f'[INFO] Invalid match: One or more column names exceed {max_length} characters, retrying...')
                    continue
                elif len(columns) > max_cols:
                    print(f'[INFO] Invalid match: Number of columns exceed {max_length} columns, retrying...')
                    continue
                elif str.lower(columns[-1]) != 'content':
                    print(f'[INFO] Invalid match: Last column is not "Content", retrying...')
                    continue

                columns = list(map(lambda x: '<'+x+'>', columns))
                formatted_columns = ' '.join(columns)
                print(f'[INFO] Columns successful generated: {formatted_columns}')
                return formatted_columns
            
            print(f"[INFO] No match found on attempt {attempt}")
        
        raise MaxRetriesError(f'[ERROR] No matches for columns found after {max_retries} attempts')
    
    def parse_log(self, input_dir, output_dir, log_file):
        """        
        Description:
        This function parses the log file using the Drain log parser and generates structured output in the specified output directory.
        
        Input:
        - input_dir: The directory containing the input log file (str)
        - output_dir: The directory to save the parsed output (str)
        - log_file: The name of the log file to be parsed (str)
        
        Output: None
        """
        file_path_full = input_dir + log_file
        columns = self.get_columns(file_path_full)
        regex = [
            r'blk_(|-)[0-9]+', # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
        ]

        st = 0.5
        depth = 4

        try:
            print('[INFO] Attempting to parse file...')
            parser = Drain.LogParser(columns, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex)
            parser.parse(log_file)
        except:
            print('[ERROR] An error has occurred during parsing.')
        else:
            print('[INFO] File parsed successfully')





if __name__ == "__main__":
    lp = LogParser(model='Llama3.1', prompt_method='fewshot')
    lp.parse_log(input_dir='../data/raw/', output_dir='result/', log_file='Windows_2k.log')
