import streamlit as st

def main():
    st.set_page_config(page_title='Ask yout CSV')
    st.header('Ask your Csv')
    
    user_csv = st.file_uploader('Upload your CSV file', type='csv')
    
    
if __name__=='__main__':
    main()