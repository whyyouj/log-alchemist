import streamlit as st
from PIL import Image, ImageEnhance
import streamlit.components.v1 as components
import logging
import base64
import time
from version_temp.lang_graph.lang_graph import Graph
from version_temp.python_agent.python_ai import Python_Ai 
from version_temp.regular_agent.agent_ai import Agent_Ai
import pandas as pd
import tempfile
import asyncio
import os, sys
import re
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.date_parser import combine_datetime_columns

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20
PANDAS_LLM = 'jiayuan1/llm2'
GENERAL_LLM = "jiayuan1/nous_llm"

st.set_page_config(
    page_title="Vantage AI",
    page_icon="imgs/vantage_logo.png",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": """
            ## Vantage AI
            #### The AI Assistant that performs log analysis

            **Contact Vantage Point Security:**  
            **https://vantagepoint.sg**

        """
    }
)

def apply_css():
    # Insert custom CSS for glowing effect
    st.markdown(
        """
        <style>
        /* Custom sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #f0f5f7;  /* 50% transparency */
            color: white;
            padding: 10px;
            border-radius: 15px;
        }

        [data-testid="stSidebar"] {
            color: #66CCFF;  /* Accent color for titles */
        }

        section[data-testid="stSidebar"] {
            width: 250px !important;
        }

        [data-testid="stExpander"]:hover details {
            border-style: none;
        }

        [data-testid="stExpander"]:hover {
            width: 100%;
            height: auto;
            border-radius: 20px;
            background-color: transparent;
            box-shadow: 
                0 0 0.5em rgba(0, 170, 255, 0.2), /* Much lighter */
                0 0 0.8em rgba(0, 170, 255, 0.4), /* Much lighter */
                0 0 1.0em rgba(0, 170, 255, 0.6), /* Slightly lighter */
                0 0 1.2em rgba(0, 170, 255, 0.8);   /* Keep this as is for contrast */
            border:0px;
            transition: transform 0.3s ease, box-shadow 0.5s ease, margin 0.3s ease; /* Smooth transition for hover */
        }
        
        [data-testid="stChatInput"]:hover{
            box-shadow: 
                0 0 0.5em rgba(0, 170, 255, 0.2), /* Much lighter */
                0 0 0.8em rgba(0, 170, 255, 0.4), /* Much lighter */
                0 0 1.0em rgba(0, 170, 255, 0.6), /* Slightly lighter */
                0 0 1.2em rgba(0, 170, 255, 0.8);   /* Keep this as is for contrast */
        }

        [data-testid="stChatInput"]{
            border-width: 1px;
            border-style: solid;
            border-color: #55B2FF;
        }

        .cover_glow {
            width: 100%;
            height: auto;
            border-radius: 15px;   /* Rounded corners for the image */
            box-shadow: 
                0 0 0.7em rgba(0, 170, 255, 0.6),
                0 0 1.2em rgba(0, 170, 255, 0.8),
                0 0 1.5em rgba(0, 170, 255, 1),
                0 0 1.8em rgba(0, 170, 255, 1.2);
            transition: transform 0.3s ease, box-shadow 0.5s ease; /* Smooth transition for hover */
        }

        #backtotop {
            background-color: #155a8a; /* Darker blue on hover */
            color: white; /* Text color */
            border: none; /* No border */
            border-radius: 50px; /* Rounded corners */
            padding: 10px 20px; /* Vertical and horizontal padding */
            font-size: 15px; /* Font size */
            cursor: pointer; /* Pointer cursor */
            transition: background-color 0.3s ease; /* Smooth transition */
            width: 15%; /* Full width */
        } 

        .aboutheader {
            background-color: #2a7bb5; /* Darker blue on hover */
            color: white; /* Text color */
            border-style: solid;
            border-width: 3px;
            border-radius: 30px; /* Rounded corners */
            padding: 10px 20px; /* Vertical and horizontal padding */
            font-size: 25px; /* Font size */
            cursor: pointer; /* Pointer cursor */
            transition: background-color 0.3s ease; /* Smooth transition */
            display: inline-block;
            margin: 10px 0px 10px 0px;
        }

        .aboutcontent {
            padding: 2px 20px;
        }
        </style>""",
        unsafe_allow_html=True,
    )

def img_to_base64(image_path):
    try:
        with open(image_path, 'rb') as file:
            return base64.b64encode(file.read()).decode()
    except Exception as e:
        logging.error(f"error converting image: {str(e)}")
        return None

def st_title():
    img_path = "imgs/vantage_logo.png"
    img_base64 = img_to_base64(img_path)

    st.markdown(f'''
        <div id='topsection' style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="data:image/png;base64,{img_base64}" style="width:50px; height:50px; margin-right: 10px;"/>
            <h1 style="display: inline; color: #020024">Vantage AI</h1>
        </div>
        ''', unsafe_allow_html=True)

def st_sidebar():
    img_path = "imgs/chatbot_logo.webp"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f"""
            <div style='padding: 20px;'>  <!-- Adjust padding as needed -->
                <img src='data:image/png;base64,{img_base64}'; class = 'cover_glow'/>
            </div>  
            """,
            unsafe_allow_html=True
        )
    
    st.sidebar.markdown("""
    <style>
    /* Style for sidebar buttons */
    .element-container:has(#chat) + div button
    {
        background-color: #1f77b4; /* Primary blue color */
        opacity: 0.6;
        color: white; /* Text color */
        border: none; /* No border */
        border-radius: 50px; /* Rounded corners */
        padding: 10px 20px; /* Vertical and horizontal padding */
        font-size: 15px; /* Font size */
        cursor: pointer; /* Pointer cursor */
        transition: background-color 0.3s ease; /* Smooth transition */
        width: 100%; /* Full width */
    }

    .element-container:has(#chat) + div button:hover {
        background-color: #155a8a; /* Darker blue on hover */
        box-shadow: 0 0 10px rgba(0, 170, 255, 0.5), 
            0 0 20px rgba(0, 170, 255, 0.7), 
            0 0 30px rgba(0, 170, 255, 1); /* Glow effect */
    }
    
    .element-container:has(#chat1) + div button {
    background-color: #155a8a; /* Darker blue on hover */
    box-shadow: 0 0 10px rgba(0, 170, 255, 0.5), 
        0 0 20px rgba(0, 170, 255, 0.7), 
        0 0 30px rgba(0, 170, 255, 1); /* Glow effect */
         color: white; /* Text color */
        border: none; /* No border */
        border-radius: 50px; /* Rounded corners */
        padding: 10px 20px; /* Vertical and horizontal padding */
        font-size: 15px; /* Font size */
        cursor: pointer; /* Pointer cursor */
        transition: background-color 0.3s ease; /* Smooth transition */
        width: 100%; /* Full width */
    }
    
    .element-container:has(#remove_chat) + div button {
    background-color: #ff4c4c; /* Primary red color */
    opacity: 0.6;
    color: white; /* Text color */
    border: none; /* No border */
    border-radius: 50px; /* Rounded corners */
    padding: 10px 20px; /* Vertical and horizontal padding */
    font-size: 15px; /* Font size */
    cursor: pointer; /* Pointer cursor */
    transition: background-color 0.3s ease; /* Smooth transition */
    width: 100%; /* Full width */
    }

    .element-container:has(#remove_chat) + div button:hover {
        background-color: #c0392b; /* Darker red on hover */
        box-shadow: 0 0 10px rgba(255, 77, 77, 0.5), 
                    0 0 20px rgba(255, 77, 77, 0.7), 
                    0 0 30px rgba(255, 77, 77, 1); /* Glow effect */
    }
    </style>
    """, unsafe_allow_html=True)

    # Use the hidden span to apply styling

    # Sidebar buttons

    if st.session_state.mode == "Chat with VantageAI":
        chat_id = "chat1"
        upload_id = "chat"
        about_id = "chat"
    elif st.session_state.mode == "Upload File":
        chat_id = "chat"
        upload_id = "chat1"
        about_id = "chat"
    else:
        chat_id = "chat"
        upload_id = "chat"
        about_id = "chat1"

    st.sidebar.markdown(f'<span id={about_id}></span>', unsafe_allow_html=True)
    if st.sidebar.button("About"):
        st.session_state.mode = "About"
        st.session_state.button = True
    
    st.sidebar.markdown(f'<span id={chat_id}></span>', unsafe_allow_html=True)
    if st.sidebar.button("My Chat", key = "ai"):
        st.session_state.mode = "Chat with VantageAI"
        st.session_state.button = True

    st.sidebar.markdown(f'<span id={upload_id}></span>', unsafe_allow_html=True)
    if st.sidebar.button("Upload File", key = "up"):
        st.session_state.mode = "Upload File"
        st.session_state.button = True

    st.sidebar.markdown('<span id="remove_chat"></span>', unsafe_allow_html=True)
    if st.sidebar.button("Restart Chat"):
        reset_session_state()
        st.session_state.button = True

    # Load and display image with glowing effect
    img_path = "imgs/vantagepoint_logo.png"
    img_base64 = img_to_base64(img_path)
    if img_base64:
        st.sidebar.markdown(
            f"""
            <div style='padding: 3em;'>
            </div>
            <hr>
            <img src='data:image/png;base64,{img_base64}' style='position: relative; bottom: 0; left: 0; '/>
            """,
            unsafe_allow_html=True
        )

def st_chatpage():
    main_col1, main_col2 = st.columns([1, 2])
    with main_col1:
        df_option = st.selectbox(
            "Select a log to query",
            st.session_state.csv_filepaths.keys(),
            index=None,
            placeholder="Select a log to query",
            label_visibility='collapsed'
        ) 

        if df_option != st.session_state.selected_df:
            update_selected_log(df_option)
    
    st.markdown(
        """
        <style>
            .stChatMessage.st-emotion-cache-1c7y2kd.eeusbqq4 {
                flex-direction: row-reverse; /* Align children to the right */
                text-align: right; /* Align text to the right */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for message in st.session_state.history[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
        output(message=message)

    if chat_input := st.chat_input("Ask a question"):

        st.session_state.history.append({"role": "user", "content": chat_input})
        output(message=st.session_state.history[-1])

        try:
            run_async_task(chat_input)
        except Exception as e:
            print(e)
            st.write("Please rephrase your question or restart the chat.")

    st.markdown(
        """ 
        <a target="_self" href="#topsection">
            <button id="backtotop">
                Back to Top
            </button>
        </a>
        """, 
        unsafe_allow_html=True
    ) 

def st_aboutpage():
    st.markdown("""
                <div class='aboutheader'>
                <b>What is Vantage AI?</b>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""
                <div class='aboutcontent'>

                **Vantage AI is a multifunctional chatbot with a focus on log analysis!**  
                
                **Timely insights from system, audit, and transaction logs are essential for maintaining system health, troubleshooting issues, and ensuring security and compliance. Audit logs capture critical system access events, transaction logs record specific application activities, and system logs monitor general performance and errors. Analysing these logs manually is time-consuming and complex, particularly as log data volumes grow.**

                **Vantage AI enables users to upload their own audit, transaction, and system logs, and quickly query, analyse, and interpret them through natural language interaction. Hence, Vantage AI can streamline troubleshooting, enhance audit reporting, and allow even non-technical users to investigate issues independently, automatically identifying patterns, providing insights, and suggesting solutions.**
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
                <div class='aboutheader'>
                <b>Navigating around Vantage AI</b>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""
                <div class='aboutcontent'>

                **:arrow_forward: Click on **:blue[My Chat]** in the sidebar to chat with Vantage AI.**  

                **:arrow_forward: Click on **:blue[Upload File]** in the sidebar to upload and manage your log files.**  

                **:arrow_forward: Click on **:red[Restart Chat]** in the sidebar to restart chat with Vantage AI.**  
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
                <div class='aboutheader'>
                <b>How to use Vantage AI</b>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""
                <div class='aboutcontent'>

                **:arrow_forward: Upload your logs by selecting them from your files or inputting an absolute folder path.**

                **:arrow_forward: Using the provided dropdown at the top of the chat, select the log you wish to query and analyse.**

                **:arrow_forward: :rainbow[Query the log you selected!] Vantage AI is able to answer questions on the selected log, provide summaries, analyse for anomalies, and even plot graphs for data visualisation!**

                **:arrow_forward: :rainbow[You may also input generic queries unrelated to your logs!] Vantage AI will respond to them like a regular chatbot!**
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
                <div class='aboutheader'>
                <b>Remarks</b>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""
                <div class='aboutcontent'>

                **:arrow_forward: Vantage AI processes suitable datetime columns, so there might be changes in the datetime columns of your logs.**

                **:arrow_forward: The size limit per log file is 50MB. Log files which exceed the size limit are not accepted.**

                **:arrow_forward: The response time of Vantage AI varies according to the complexity of the query and the size of the selected log.**

                **:arrow_forward: Try to provide more context in your queries to allow Vantage AI to generate better responses.**
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("""
                <div class='aboutheader'>
                <b>Contact Us</b>
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""
                <div class='aboutcontent'>

                **Website: https://vantagepoint.sg**
                </div>
                """, unsafe_allow_html=True)

def st_fileuploader():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Uploaded Files')
        st.button('Clear', on_click=clear_files)
        if len(st.session_state.csv_filepaths) > 0: #len(st.session_state.filepaths) > 0 or len(st.session_state.csv_filepaths) > 0:
            all_files = list(st.session_state.csv_filepaths.keys()) #list(st.session_state.filepaths.keys()) + list(st.session_state.csv_filepaths.keys())
            files_df = pd.DataFrame({'No.': range(1, len(all_files) + 1), 'File': all_files})
            st.dataframe(files_df, hide_index=True)
        else:
            st.text('No uploaded files yet!')

    with col2:
        display_file_uploader()

def display_file_uploader():
    with st.form(key = "fileupload_form"):
        #desired file types: type=['pdf', 'txt', 'log', 'docx', 'csv']
        uploaded_files = st.file_uploader("Upload your log files", type=['csv'], 
                                            accept_multiple_files=True, label_visibility='visible')
        Submit = st.form_submit_button(label = "Submit")
    if Submit:
        on_file_submit(uploaded_files)
        if len(st.session_state.csv_filepaths) > 0: #len(st.session_state.filepaths) > 0 or len(st.session_state.csv_filepaths) > 0:
            success_message = st.success(f"Upload successful!")
            time.sleep(3)
            success_message.empty()
        st.rerun()

    with st.form(key = "folderupload_form"):
        abs_folderpath = st.text_area('Input an absolute folder path')
        folder_submit = st.form_submit_button(label = "Submit")
    if folder_submit:
        on_folder_submit(abs_folderpath)
        if len(st.session_state.csv_filepaths) > 0: #len(st.session_state.filepaths) > 0 or len(st.session_state.csv_filepaths) > 0:
            success_message = st.success(f"Upload successful!")
            time.sleep(3)
            success_message.empty()
        st.rerun()
        
def on_file_submit(uploaded_files):
    filepaths = st.session_state.filepaths.copy()
    csv_filepaths = st.session_state.csv_filepaths.copy()

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            # file_path = tf.name

            if uploaded_file.name[-3:] == 'csv':
                csv_filepaths[uploaded_file.name] = tf #file_path
            else:
                filepaths[uploaded_file.name] = tf #file_path

    if filepaths != st.session_state.filepaths:
        st.session_state.filepaths = filepaths
        print('FILE PATHS: ', st.session_state.filepaths)

    if csv_filepaths != st.session_state.csv_filepaths:
        st.session_state.csv_filepaths = csv_filepaths
        print('CSV FILE PATHS: ', st.session_state.csv_filepaths)
        # update_langgraph()

def on_folder_submit(abs_folderpath):
    abs_folderpath = abs_folderpath.strip()
    if len(abs_folderpath) == 0:
        return
    
    if abs_folderpath[-1] != '/':
        abs_folderpath += '/'

    err_str = ''
    if not os.path.isabs(abs_folderpath):
        err_str += "Error: Given path is not an absolute path  \n"
    elif not os.path.isdir(abs_folderpath):
        err_str += "Error: Given path does not exist as a folder   \n"

    if len(err_str) > 0:
        err_message = st.error(err_str)
        time.sleep(7)
        err_message.empty()
    else:
        filepaths = st.session_state.filepaths.copy()
        csv_filepaths = st.session_state.csv_filepaths.copy()

        for file in os.listdir(abs_folderpath):
            abs_path = abs_folderpath + file
            #check that abs_path is an existing regular file and file is within size limit of 50MB
            if os.path.isfile(abs_path) and (os.path.getsize(abs_path) / 10**6 <= 50):
                if file.endswith('.csv'):
                    csv_filepaths[file] = abs_path
                elif not file.endswith('.DS_Store'):
                    filepaths[file] = abs_path
        
        if filepaths != st.session_state.filepaths:
            st.session_state.filepaths = filepaths
            print('FILE PATHS: ', st.session_state.filepaths)

        if csv_filepaths != st.session_state.csv_filepaths:
            st.session_state.csv_filepaths = csv_filepaths
            print('CSV FILE PATHS: ', st.session_state.csv_filepaths)
            # update_langgraph()

def clear_files():
    for file in list(st.session_state.filepaths.values()) + list(st.session_state.csv_filepaths.values()):
        if isinstance(file, tempfile._TemporaryFileWrapper):
            os.remove(file.name)
    st.session_state.filepaths = {}
    st.session_state.csv_filepaths = {}
    print('Uploaded Files Cleared')
    # update_langgraph()

def output(message):
    role = message["role"]
    avatar_image = "imgs/bot.png" if role == "assistant" else "imgs/user.png" if role == "user" else None
    
    with st.chat_message(role, avatar=avatar_image):
        #if the message from user or assistant is just a string, output it without formatting
        if isinstance(message['content'], str):
            st.write(message['content'])
            return 

        #if assistant message is a list of dict, determine whether a not to break the question down
        format = True
        if len(message['content']) == 1:
            format = False
        if format:
            st.write("Here is the break down of your question:")
        for dict in message['content']:
            if format:
                st.write(dict['qns'])
            out = dict['ans']
            if str(out).endswith(".html"):
                st.write("Here is a summary of the data!")
                print("[APP]", out)
                pattern = r"([A-Z]:(?:\\\\|\\)(?:[^\\/:*?\"<>|\r\n]+(?:\\\\|\\))*[^\\/:*?\"<>|\r\n]+\.[a-zA-Z0-9]+|(?:\/[^\/\s]+)+\/[^\/\s]+\.[a-zA-Z0-9]+)"
                html_files = re.findall(pattern, out)
                print(html_files)
                for path in html_files:
                    if os.path.exists(path):
                        # Read the file and render it in an iframe
                        with open(path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        # Display the HTML report in Streamlit
                        components.html(html_content, height=800, scrolling=True)
                
                        # Deleting temporary file after outputing
                        os.remove(path)
                    
            elif "exports/charts/" in str(out) or 'tabulated_anomalies.png' in str(out) or '.png' in str(out):
                img_base64 = img_to_base64(out)
                if img_base64:
                    st.markdown(
                            f"""
                            Here is the Chart!
                            <img src='data:image/png;base64,{img_base64}'/>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.write(f"I'm so sorry. But I am unable to show you the plotted graph.")
            else:
                st.write(out)   
        return

async def on_chat_submit(chat_input):
    """
    Handle chat input submissions and interact with the llm.

    Parameters:
    - chat_input (str): The chat input from the user.

    Returns:
    - None: Updates the chat history in Streamlit's session state.
    """

    try:
        graph = st.session_state.graph
        out = graph.run(chat_input)

        st.session_state.history.append({"role": "assistant", "content": out})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        error_message = st.error(f"AI Error: {str(e)}")
        time.sleep(3)
        error_message.empty()

def run_async_task(chat_input):
    with st.spinner("Thinking..."):
        # Run the async function within an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(on_chat_submit(chat_input))
        output(st.session_state.history[-1])

def initialize_conversation():
    assistant_message = "Hello! I am Vantage AI. How can I assist you today?"
    conversation_history = [
        {"role":"assistant", "content":[ {"qns":"Begin", "ans": assistant_message} ]}
    ]
    return conversation_history

def initialize_langgraph():
    agent = Agent_Ai(model= GENERAL_LLM)
    print('LangGraph Initialized')
    return agent

def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = initialize_conversation()
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat with VantageAI"
    if "button" not in st.session_state:
        st.session_state.button = False
    if "filepaths" not in st.session_state:
        st.session_state.filepaths = {}
    if "csv_filepaths" not in st.session_state:
        st.session_state.csv_filepaths = {}
        #default folder path: set a fixed directory from which to retrieve the logs
        default_abs_folder = os.path.abspath('../logs/Test')
        on_folder_submit(default_abs_folder)
    if "graph" not in st.session_state:
        st.session_state.graph = initialize_langgraph()
    if "selected_df" not in st.session_state:
        st.session_state.selected_df = None

def reset_session_state():
    st.session_state.history = initialize_conversation()
    st.session_state.mode = "Chat with VantageAI"
    st.session_state.selected_df = None

def update_langgraph():
    df_list = []
    for file in st.session_state.csv_filepaths.values():
        file_path = file
        if isinstance(file, tempfile._TemporaryFileWrapper):
            file_path = file.name

        df = pd.read_csv(file_path)
        date_formatted_df = combine_datetime_columns(df)
        df_list.append(date_formatted_df)

    llm = Python_Ai(model = PANDAS_LLM, df = df_list)
    pandas_llm = llm.pandas_legend_with_skill()
    graph = Graph(pandas_llm=pandas_llm, df=df_list)
    st.session_state.graph = graph
    print("LangGraph Updated")

def update_selected_log(df_option):
    df_list = []

    if df_option is not None:
        file = st.session_state.csv_filepaths[df_option]
        file_path = file
        if isinstance(file, tempfile._TemporaryFileWrapper):
            file_path = file.name

        df = pd.read_csv(file_path)
        date_formatted_df = combine_datetime_columns(df)
        df_list.append(date_formatted_df)

    llm = Python_Ai(model = PANDAS_LLM, df = df_list)
    pandas_llm = llm.pandas_legend_with_skill()
    graph = Graph(pandas_llm=pandas_llm, df=df_list)
    st.session_state.graph = graph
    st.session_state.selected_df = df_option
    print("LangGraph updated with selected log:", df_option)

def main():
    
    """
    Display the chat interface :).
    """
    
    initialize_session_state()

    st_title()
    apply_css()
    st_sidebar()

    if st.session_state.mode == "Chat with VantageAI":
        st_chatpage()           

    if st.session_state.mode == "Upload File":
        st_fileuploader()

    if st.session_state.mode == "About":
        st_aboutpage()

    if st.session_state.button:
        st.session_state.button = False
        st.rerun()


if __name__=="__main__":
    main()