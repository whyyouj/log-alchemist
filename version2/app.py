import streamlit as st
from PIL import Image, ImageEnhance
import logging
import base64
import time
from version_temp.lang_graph.lang_graph import Graph
from version_temp.python_agent.python_ai import Python_Ai 
import pandas as pd
# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
NUMBER_OF_MESSAGES_TO_DISPLAY = 20

st.set_page_config(
    page_title="Vantage Assistant",
    page_icon="imgs/vantage_logo.png",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "About": """
            ## Vantage AI Assistant
            ### Powered using Llama 3.1

            **Vantage Point Security**:https://vantagepoint.sg

            The AI Assistant that performs log analysis.
        """
    }
)

# st.title("Vantage AI")
def img_to_base64(image_path):
    try:
        with open(image_path, 'rb') as file:
            return base64.b64encode(file.read()).decode()
    except Exception as e:
        logging.error(f"error converting image: {str(e)}")
        return None

img_path = "imgs/vantage_logo.png"
img_base64 = img_to_base64(img_path)

st.markdown(f'''
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <img src="data:image/png;base64,{img_base64}" style="width:50px; height:50px; margin-right: 10px;"/>
        <h1 style="display: inline; color: #020024">Vantage AI</h1>
    </div>
    ''', unsafe_allow_html=True)



    

@st.cache_data(show_spinner=True)
def long_running_task(duration):
    time.sleep(duration)
    return "Long-running task completed"

# @st.cache_data(show_spinner = False)
# def load_and_enhance_image(image_path, enhance=True):
#     img = Image.open(image_path)
#     if enhance:
#         enhancer = ImageEnhance.Contrast(img)
#         img = enhancer(1.8)
#     return img

def display_file_uploader():
    with st.expander("**File Upload**", expanded=False):
        with st.form(key = "Form :", clear_on_submit= True):
            File = st.file_uploader(label="Upload File", type = ["logs", "csv"])
            Submit = st.form_submit_button(label = "Submit")
        if Submit:
            
            try:
                save_folder = f"{File.name}"
                success_message = st.success(f"File {File.name} uploaded successfully")
                time.sleep(3)
                success_message.empty()
            except:
                error_message = st.error(f"No File Found")
                time.sleep(3)
                error_message.empty()
                
                
def initialize_conversation():
    assistant_message = "Hello! I am Vantage AI. How can I assist you today?"
    conversation_history = [
        {"role":"assistant", "content":assistant_message}
    ]
    return conversation_history


# @st.cache_data(show_spinner=True)
def on_chat_submit(chat_input):
    """
    Handle chat input submissions and interact with the llm.

    Parameters:
    - chat_input (str): The chat input from the user.

    Returns:
    - None: Updates the chat history in Streamlit's session state.
    """
    user_input = chat_input.strip().lower()

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = initialize_conversation()

    st.session_state.conversation_history.append({"role": "user", "content": user_input})

    try:
        
        # import matplotlib.pyplot
        # import pandas as pd
        # df = pd.read_csv('../EDA/data/mac/Mac_2k.log_structured.csv')
        # import pandasai as pai
        # from langchain_community.llms import Ollama
        
        # llm = Ollama(
        #     model = "llama3.1",
        #     temperature = 0.2
        # )
        # pandas_ai_agent = pai.SmartDataframe(df, config={"llm":llm})
        # pandas_ai_agent.chat(user_input)
        #time.sleep(1)
        # graph = st.session_state.graph
        # out = graph.run(user_input)
        assistant_reply = 'exports/charts/d772a0b7-7737-410f-9464-1427a93b2a1d.png' #out


        st.session_state.conversation_history.append({"role": "assistant", "content": assistant_reply})
        st.session_state.history.append({"role": "user", "content": user_input})
        st.session_state.history.append({"role": "assistant", "content": assistant_reply})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        error_message = st.error(f"AI Error: {str(e)}")
        time.sleep(3)
        error_message.empty()
        
def initialize_session_state():
    """Initialize session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat with VantageAI"
    if "button" not in st.session_state:
        st.session_state.button = False
    if "graph" not in st.session_state:
        df = pd.read_csv("../EDA/data/mac/Mac_2k.log_structured.csv")
        llm = Python_Ai(df = df)
        pandas_llm = llm.pandas_legend()
        graph = Graph(pandas_llm=pandas_llm, df=df)
        st.session_state.graph = graph
        print("pandas legend intialise")


def main():
    
    """
    Display the chat interface :).
    """
    
    initialize_session_state()

    # print(st.session_state.mode, st.session_state.conversation_history)
    if not st.session_state.history:
        initial_bot_message = "Hello! I am Vantage AI. How can I assist you today?"
        st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
        st.session_state.conversation_history = initialize_conversation()


    # Insert custom CSS for glowing effect
    st.markdown(
        """

     <style>


        /* Custom sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0.1);  /* 50% transparency */
            color: white;
            padding: 10px;
            border-radius: 15px;
        }

        [data-testid="stSidebar"] {
            color: #66CCFF;  /* Accent color for titles */
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
            border:0px;
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
        

        </style>""",
        unsafe_allow_html=True,
    )
    img_path = "imgs/robot.png"
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
        border-radius: 5px; /* Rounded corners */
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
        border-radius: 5px; /* Rounded corners */
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
    border-radius: 5px; /* Rounded corners */
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
    
    if "mode" not in st.session_state:
        st.session_state.mode = "Chat with VantageAI"

    # mode = "Chat with VantageAI"

    if st.session_state.mode == "Chat with VantageAI":
        chat_var = "chat1"
        chat_var2 = "chat"
    else:
        chat_var = "chat"
        chat_var2 = "chat1"
    
    st.sidebar.markdown(f'<span id={chat_var}></span>', unsafe_allow_html=True)
    if st.sidebar.button("Chat with Vantage AI", key = "ai"):
        # chat_var = "chat1"
        # chat_var2 = "chat"
        st.session_state.mode = "Chat with VantageAI"
        st.session_state.button = True

    st.sidebar.markdown(f'<span id={chat_var2}></span>', unsafe_allow_html=True)
    if st.sidebar.button("Upload File", key = "up"):
        # chat_var = "chat"
        # chat_var2 = "chat1"
        st.session_state.mode = "Upload File"
        st.session_state.button = True
 

    st.sidebar.markdown('<span id="remove_chat"></span>', unsafe_allow_html=True)
    if st.sidebar.button("Restart Chat"):
        st.session_state.history = []
        st.session_state.conversation_history = []
        initial_bot_message = "Hello! I am Vantage AI. How can I assist you today?"
        st.session_state.history.append({"role": "assistant", "content": initial_bot_message})
        st.session_state.conversation_history = initialize_conversation()
        st.session_state.mode = "Chat with VantageAI"
        chat_var = "chat1"
        chat_var2 = "chat"
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
        

    if st.session_state.mode == "Chat with VantageAI":
        
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
                    role = message["role"]
                    avatar_image = "imgs/ai.png" if role == "assistant" else "imgs/person.png" if role == "user" else None
                    with st.chat_message(role, avatar=avatar_image):
                        if "exports/charts/" in message['content']:
                            img_base64 = img_to_base64(message['content'])
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
                            st.write(message["content"])

        if chat_input := st.chat_input("Ask a question:"):

            role = "user"
            avatar_image = "imgs/ai.png" if role == "assistant" else "imgs/person.png" if role == "user" else None
            with st.chat_message(role, avatar=avatar_image):
                        st.write(chat_input)
                        
            with st.spinner("thinking..."):
                on_chat_submit(chat_input)
            role = st.session_state.history[-1]['role']
            content = st.session_state.history[-1]['content']
            avatar_image = "imgs/ai.png" if role == "assistant" else "imgs/person.png" if role == "user" else None
            with st.chat_message(role, avatar=avatar_image):
                if "exports/charts/" in message['content']:
                    img_base64 = img_to_base64(message['content'])
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
                    st.write(message["content"])
                    
            
        # chat_input = st.chat_input()
        # spinner_placeholder = st.empty()
        # with st.spinner("thinking..."):
        #     if chat_input:
        #         spinner_placeholder = st.empty()
        #         with spinner_placeholder:
                    
        #                 on_chat_submit(chat_input)
            
            # Display chat history
            
                # for message in st.session_state.history[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
                #     role = message["role"]
                #     avatar_image = "imgs/ai.png" if role == "assistant" else "imgs/person.png" if role == "user" else None
                #     st.markdown(
                #     """
                #     <style>
                #         .stChatMessage.st-emotion-cache-1c7y2kd.eeusbqq4 {
                #             flex-direction: row-reverse; /* Align children to the right */
                #             text-align: right; /* Align text to the right */
                #         }
                #     </style>
                #     """,
                #         unsafe_allow_html=True,
                #     )
                #     with st.chat_message(role, avatar=avatar_image):
                #         st.write(message["content"])

    if st.session_state.mode =="Upload File":
        display_file_uploader()

    if st.session_state.button:
        st.session_state.button = False
        st.rerun()


if __name__=="__main__":
    main()