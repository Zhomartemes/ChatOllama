import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import chromadb
from chromadb.config import Settings
import uuid

logging.basicConfig(level=logging.INFO)

# Initialize Ollama Embeddings
ollama_embedding = OllamaEmbeddings(
    model="all-minilm",
)

# Specify the directory where ChromaDB will persist data
persist_directory = "./chroma_db"  # Adjust the path as needed

# Initialize ChromaDB client with persistence directory using Settings
settings = Settings(
    persist_directory=persist_directory
)

client = chromadb.Client(settings)  # Pass settings to client

# Create or get the collection for embeddings with persistent storage
embeddings_collection = client.get_or_create_collection("embeddings")

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Function to stream chat
def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

# Function to generate embeddings and store them in ChromaDB
def generate_embeddings(question, response):
    try:
        logging.info("Generated embeddings")
        inputs = [question, response]
        embeddings = ollama_embedding.embed_documents(inputs)
        embeddings_dict = {
            "request_embedding": embeddings[0],
            "response_embedding": embeddings[1],
        }

        embeddings_data = {
            "question": question,
            "response": response,
            "request_embedding": embeddings_dict["request_embedding"],
            "response_embedding": embeddings_dict["response_embedding"],
            "timestamp": time.time(),
        }

        # Generate unique IDs for each document
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        embeddings_collection.add(
            ids=ids,
            documents=[question, response],
            metadatas=[{"type": "request"}, {"type": "response"}],
            embeddings=[embeddings_dict["request_embedding"], embeddings_dict["response_embedding"]]
        )
        logging.info("Embeddings stored in ChromaDB")

        return embeddings_dict
    except Exception as e:
        logging.error(f"Error while generating embeddings: {str(e)}")
        raise e

# Main function to run the Streamlit app
def main():
    # Apply custom CSS for appearance


    st.title("Chat with LLMs - Interactive Model Chat")
    logging.info("App started")

    model = st.sidebar.selectbox("Choose a model", ["llama3.2"])
    logging.info(f"Model selected: {model}")

    if prompt := st.chat_input("Ask your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Assistant is typing..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in
                                    st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        # Removed duration display

                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_message})
                        logging.info(f"Response: {response_message}")

                        embeddings = generate_embeddings(prompt, response_message)

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
