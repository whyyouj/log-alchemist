import pandas as pd
import datetime
import re

def break_down_date_component(date_str):
    """
    Parses a date string and returns its components (year, month, day) based on various formats.

    Function Description:
    This function is designed to parse and extract components (year, month, day) from a string
    representing a date in various formats. The function handles:
      - Full dates (e.g., "2007-08-11").
      - Single-day representations (e.g., "11").
      - Month abbreviation with day (e.g., "Aug 11").
      - Day followed by month abbreviation (e.g., "11 Aug").
      - Month-only representations (e.g., "Jul" or "July").
    
    The function first tries to convert the input string into a datetime object.
    If that fails, it uses regular expressions to handle other date-like formats.

    Input:
    - date_str (str): A string representing a date or a part of a date in various formats.
    
    Output:
    - dict: A dictionary containing the following keys if the date components are parsed successfully:
        - 'year' (int): The year component of the date.
        - 'month' (int): The month component of the date (1-12).
        - 'day' (int): The day component of the date (1-31).
    - None: If the input doesn't match any recognizable date format.
    """

    # Try to convert it to a datetime object first (handles full dates like '2007-08-11')
    try:
        full_date = pd.to_datetime(date_str, errors='raise')
        return {
            'year': full_date.year,
            'month': full_date.month,
            'day': full_date.day
        }
    except (ValueError, TypeError):
        pass

    # Check if it's just a day (a single digit or two digits)
    if re.match(r'^\d{1,2}$', date_str):
        return {'day': int(date_str)}

    # Check if it's a month abbreviation and a day (e.g., "Aug 11")
    month_day_match = re.match(r'^([A-Za-z]+)\s+(\d{1,2})$', date_str)
    if month_day_match:
        month_str = month_day_match.group(1)
        day = int(month_day_match.group(2))
        month = pd.to_datetime(month_str, format='%b', errors='coerce').month
        return {'month': month, 'day': day}

    # Check if it's a day followed by a month (e.g., "11 Aug")
    day_month_match = re.match(r'^(\d{1,2})\s+([A-Za-z]+)$', date_str)
    if day_month_match:
        day = int(day_month_match.group(1))
        month_str = day_month_match.group(2)
        month = pd.to_datetime(month_str, format='%b', errors='coerce').month
        return {'month': month, 'day': day}
    
    # Check if the date_str is just a month (either abbreviation like 'Jul' or full name like 'July')
    month_match = re.match(r'^([A-Za-z]+)$', date_str)
    if month_match:
        month_str = month_match.group(1)
        month = pd.to_datetime(month_str, format='%b', errors='coerce').month
        
        if pd.isna(month):  # If it's not a valid abbreviated month, try full month name
            month = pd.to_datetime(month_str, format='%B', errors='coerce').month
        
        if month:  # Return only the month if matched
            return {'month': month}

    # If no match, return None or raise an error
    return None

def combine_datetime_columns(df, default_year=datetime.datetime.now().year):
    """
    Combines datetime-related components in a DataFrame into a single 'Datetime' column.

    Function Description:
    This function identifies columns in a DataFrame that contain datetime-related information 
    (e.g., year, month, day, time) and combines them into a single 'Datetime' column. 
    It also attempts to handle partial date components from a "Date" column, using the 
    `break_down_date_component` function to extract year, month, and day. If any of these 
    components are missing, a default year can be provided. The combined 'Datetime' column 
    is converted into a proper datetime object. The original datetime-related columns are 
    dropped from the DataFrame after successful conversion.

    Input:
    - df (pandas.DataFrame): A DataFrame containing date-related columns (e.g., year, month, day, etc.).
    - default_year (int, optional): A default year to use if the year component is missing. Defaults to the current year.

    Output:
    - df (pandas.DataFrame): The modified DataFrame with a 'Datetime' column containing the 
      combined datetime information. Original datetime-related columns are removed.
    
    Note:
    - If no datetime-related columns are found, the function returns the DataFrame as is.
    - If the conversion to a datetime object fails, an error message is printed.
    """

    # Define a list of common datetime-related column names
    datetime_keywords = ["year", "month", "day", "date", "time", "hour", "minute", "second"]

    # Identify columns that likely contain datetime information
    datetime_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in datetime_keywords)]
    
    if not datetime_cols:
        print("No datetime components found.")
        return df

    #keep copy of the original df to return in case conversion fails
    df_orig = df.copy()

    # Check if "Date" column already contains full or partial date components
    for col in df.columns:
        if 'date' in col.lower():
            # Attempt to break down the 'Date' column
            df[col] = df[col].astype(str)
            date_parts = df[col].apply(break_down_date_component)
            df['Year'] = [date.get('year', default_year) for date in date_parts]
            df['Month'] = df["Month"] if "Month" in df.columns else [date.get('month', None) for date in date_parts]
            df['Day'] = df["Day"] if "Day" in df.columns else [date.get('day', None) for date in date_parts]

            # Now combine the components into the 'Datetime' column
            if 'Time' in df.columns:
                df['Datetime'] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str) + ' ' + df['Time'].astype(str)
            else:
                df['Datetime'] = df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str)

            if "Year" not in datetime_cols:
                datetime_cols.append("Year")
            if "Month" not in datetime_cols:
                datetime_cols.append("Month")
            if "Day" not in datetime_cols:
                datetime_cols.append("Day")

    # Convert the combined "Datetime" column into a proper datetime format
    try:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        # Drop the original datetime-related columns after successful conversion
        cols_to_drop = datetime_cols
        if 'Datetime' in cols_to_drop:
            cols_to_drop.remove('Datetime')
        df.drop(columns=cols_to_drop, inplace=True)
        return df
    except Exception as e:
        print(f"Error converting to datetime, please check the format of the combined column: {e}")

    return df_orig

if __name__ == '__main__':
    file_name = "Windows_2k.log_structured.csv"
    df = pd.read_csv(f"../logs/Windows/{file_name}")
    path = f"../logs/Test/{file_name}"
    combine_datetime_columns(df).to_csv(path, index = False)