import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import BedrockChat
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import re

def download_file_from_s3(bucket_name, key, local_path):
    """Download a file from S3 to a local path."""
    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, key, local_path)
        st.write(f"Downloaded {key} from S3 to {local_path}")
    except NoCredentialsError:
        st.write("Error: No AWS credentials found.")
    except PartialCredentialsError:
        st.write("Error: Incomplete AWS credentials found.")
    except ClientError as e:
        st.write(f"ClientError: {e.response['Error']['Message']}")
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return "\n\n".join([page.page_content for page in pages])

def load_data_from_s3(bucket_name, prefix):
    """Load and concatenate text from all PDF files in an S3 bucket."""
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        pdf_files = [content['Key'] for content in response.get('Contents', []) if content['Key'].endswith('.pdf')]
        
        all_extracted_text = ""
        with ThreadPoolExecutor(12) as executor:
            local_paths = []
            for pdf_file in pdf_files:
                local_path = os.path.join("temp", os.path.basename(pdf_file))
                download_file_from_s3(bucket_name, pdf_file, local_path)
                local_paths.append(local_path)
            
            pdf_texts = executor.map(extract_text_from_pdf, local_paths)
            all_extracted_text = "\n\n".join(pdf_texts)
        
        for path in local_paths:
            os.remove(path)
        
        st.write("Processing of all PDFs done")
        return all_extracted_text
    
    except Exception as e:
        st.write(f"An error occurred while loading data from S3: {e}")

def split_pdf_text(docs):
    """Split large document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        separators=['\n\n', '\n'],
        chunk_overlap=150
    )
    doc = Document(page_content=docs)
    splits = text_splitter.split_documents([doc])
    print("Split length:", len(splits))
    return splits

def generate_embeddings_and_vector_db(splits):
    """Generate embeddings and create vector database."""
    try:
        if not splits:
            raise ValueError("No document splits to process")

        bedrock_client = boto3.client(service_name="bedrock-runtime")
        embedding = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=bedrock_client)
        
        persist_directory = os.path.join('docs/chroma', ''.join(random.choices(string.ascii_letters + string.digits, k=20)))
        
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        
        print("Collection size:", vectordb._collection.count())
        return vectordb
    
    except Exception as e:
        st.write(f"An error occurred while generating embeddings: {e}")

def generate_prompt_template():
    """Generate prompt template for QA chain."""
    template = """## Zensar Policy Assistant

**Instructions:**

You are a helpful and informative assistant for Zensar employees, providing guidance on HR policies and procedures. Analyze the user's `{question}` within the provided `{context}` and follow these instructions before formulating your response:

1. **Relevance:** If the {question} does not have any relevance to the {context}, then politely inform the user and ask them to ask questions specifically about Zensar policies. 
2. **Sensitive Information:** If the `{question}` or `{context}` contains sensitive or confidential information, advise the user not to share such information and explain the importance of data security. 
3. **Security & Compliance:** If the `{question}` or `{context}` involves organization-level security and compliance issues, advise the user to avoid any actions that might violate policies and suggest contacting the security team for further guidance.
4. **Policy Adherence:** If the `{question}` or `{context}` suggests a potential breach of standard policies, advise the user against such actions and explain the potential consequences.
5. **Greetings:** If the `{question}` is a simple greeting like "Hi" or "Hello," greet the user in a friendly manner and ask for their specific question.
6. **Direct Questions:** For direct questions about policies, provide a clear and concise answer without mentioning your internal analysis or the type of question.
7. **Yes/No Questions:** For yes/no questions or those starting with "Is," respond with "Yes" or "No" followed by a brief explanation.
8. **Fill-in-the-blanks:** If the `{question}` or `{context}` contains blanks (indicated by "_" or "-") or requires filling in missing information, provide the missing information directly without restating the question.

**Remember, your final response should only contain the answer to the user's query. Do not include any internal workings or analysis.**

**Context:**

{context}

**User Query:**

{question}
"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question"], template=template)
    return QA_CHAIN_PROMPT

def query_retrieval_pipeline(vectordb, QA_CHAIN_PROMPT, question):
    """Run query retrieval pipeline."""
    try:
        llm = BedrockChat(model_id="amazon.titan-text-express-v1")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(),
            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
            memory=memory,
        )
        result = qa_chain({"question": question})
        return result["answer"]
    
    except Exception as e:
        st.write(f"An error occurred during query retrieval: {e}")

def chat_actions():
    """Perform actions based on user input."""
    user_question = st.session_state.get("user_question", "")

    if len(re.sub('[^A-Za-z]+', '', str(user_question))) == 0:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_question,
            "is_user": True
        })
        result = f'''I'm sorry, but your question '{user_question}' is not clear. Please provide more context or details so I can assist you better. If you have any questions related to Zensar policies, feel free to ask, and I'll be happy to help.'''
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": result,
            "is_user": False
        })
    else:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": user_question,
            "is_user": True
        })
        result = query_retrieval_pipeline(st.session_state["vectordb"],
                                          st.session_state["QA_CHAIN_PROMPT"],
                                          user_question)
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": result,
            "is_user": False
        })

def refresh_chat():
    """Refresh chat history."""
    st.session_state["chat_history"] = []
    st.experimental_rerun()

def main():
    """Main function."""
    st.set_page_config(page_title="HR Policy Chatbot", page_icon=":robot_face:")

    # Initialize once
    if "vectordb" not in st.session_state:
        bucket_name = "zen-policy"
        prefix = "documents/"

        docs = load_data_from_s3(bucket_name, prefix)
        if not docs:
            st.error("Failed to load documents from S3.")
            return
        
        splits = split_pdf_text(docs)
        vectordb = generate_embeddings_and_vector_db(splits)
        QA_CHAIN_PROMPT = generate_prompt_template()

        st.session_state["vectordb"] = vectordb
        st.session_state["QA_CHAIN_PROMPT"] = QA_CHAIN_PROMPT

    # Display styled title
    header = st.container()
    with header:
        st.markdown("""
        <img style='width:50%;position:relative;margin-left:25%' src='https://rpgnet.sharepoint.com/sites/OneRPG/SPFx/LogoSlider/images/logo-4.png'>
        """, unsafe_allow_html=True)
        st.markdown("<style> h1 { color: #3f51b5; text-align: center; } </style>", unsafe_allow_html=True)
    
    st.title("HR Policy Chatbot :robot_face:")

    # Placeholder for chat input (at the top)
    input_placeholder = st.empty()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Refresh Chat"):
        refresh_chat()

    # User input
    with input_placeholder.container():
        user_question = st.chat_input("Your Question:", on_submit=chat_actions, key="user_question")

    # Process question and display results
    if user_question:
        with st.spinner("Processing your question..."):
            time.sleep(2)  # Simulate processing time

            for message in st.session_state["chat_history"]:
                with st.chat_message(name=message["role"]):
                    st.write(message["content"])

    else:
        st.warning("Please enter a question.")

    # Footer content
    footer = st.container()
    with footer:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<h3>Powered by <span style='color:aqua;font-size:bold;'>Zensar<span><h3>", unsafe_allow_html=True)
    st.markdown("""
    <style>
        .stApp {
            font-family: Arial, sans-serif;
            width:80%;
            border:10px solid grey;
            margin:0 auto;
        }
        .stTextInput > div > div > input {
            border-radius: 10px;
            padding: 10px;
        }
        .stTextArea > div > div > textarea {
            border-radius: 10px;
            padding: 10px;
        }
        .stButton > button {
            position: relative;
        }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
