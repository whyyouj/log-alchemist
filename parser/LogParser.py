from ColumnGetter import ColumnGetter
from logparser import Drain
import re

# Custom exception for column generation failures
class MaxRetriesError(Exception):
    pass

class LogParser:
    def __init__(self, model, prompt_method):
        """
        Initializes a LogParser instance with specified model and prompt method.

        Function Description:
        Creates a new LogParser instance that will use the specified language model
        and prompting strategy for parsing log files and generating column names.

        Input:
        - model (str): Name of the language model to use (e.g., 'Llama3.1')
        - prompt_method (str): Method for prompting ('default' or 'fewshot')

        Output:
        - None (initializes instance attributes)

        Note:
        - Instance will use default model behavior if invalid model name provided
        - Prompt method defaults to 'default' if invalid method specified
        """
        self.model = model
        self.prompt_method = prompt_method

    def read_file(self, path):
        """
        Reads and returns the contents of a log file.

        Function Description:
        Opens and reads a log file from the specified path, concatenating all lines
        with newline characters preserved to maintain the original format.

        Input:
        - path (str): Full path to the log file to be read

        Output:
        - log (str): Complete contents of the log file as a single string

        Note:
        - Returns empty string if file doesn't exist or can't be read
        - Preserves original line endings and formatting
        """
        log = ''
        with open(path, mode='r') as f:
            for line in f:
                log += (line + '\n')
        return log
    
    def get_columns(self, path, max_retries=5, max_length=20, max_cols = 10):
        """
        Generates and validates column names for log data.

        Function Description:
        Attempts to generate appropriate column names for the log data by:
        1. Reading the log file
        2. Using ColumnGetter to generate column names
        3. Validating the generated columns against quality criteria
        4. Formatting the validated columns for the parser

        Input:
        - path (str): Path to the log file
        - max_retries (int): Maximum attempts to generate valid columns (default=5)
        - max_length (int): Maximum allowed length for column names (default=20)
        - max_cols (int): Maximum allowed number of columns (default=10)

        Output:
        - formatted_columns (str): Space-separated string of validated column names
        enclosed in angle brackets

        Note:
        - Raises MaxRetriesError if valid columns not generated within max_retries
        - Column names are sanitized to contain only letters and underscores
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
                # Sanitize and validate column names
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
        Parses a log file using the Drain algorithm.

        Function Description:
        Processes the specified log file using the Drain log parsing algorithm:
        1. Generates appropriate column names
        2. Configures regex patterns for parsing
        3. Initializes and runs the Drain parser
        4. Saves structured output to the specified directory

        Input:
        - input_dir (str): Directory containing the input log file
        - output_dir (str): Directory where parsed results will be saved
        - log_file (str): Name of the log file to parse

        Output:
        - None (generates files in output_dir)

        Note:
        - Creates output directory if it doesn't exist
        - Prints error message but doesn't raise exception if parsing fails
        """
        file_path_full = input_dir + log_file
        columns = self.get_columns(file_path_full)
        # Regular expressions for identifying common log patterns
        regex = [
            r'blk_(|-)[0-9]+', # block id
            r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
            r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
        ]
        # Drain parser parameters
        st = 0.5    # Similarity threshold for log grouping
        depth = 4   # Maximum depth of parsing tree

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
