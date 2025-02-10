# Medical Chatbot using Retrieval-Augmented Generation (RAG)

This project implements a **Retrieval-Augmented Generation (RAG)**-based **Medical Chatbot** that can answer health-related queries using a **pretrained language model** and **vector search**. It processes medical PDFs, embeds text into a **vector database**, and retrieves relevant information to generate responses.

## Features
- **Document Processing**: Loads and processes medical PDFs.
- **Text Chunking**: Splits text into smaller, retrievable chunks.
- **Embeddings**: Converts text into numerical representations using **Hugging Face embeddings**.
- **Vector Search**: Stores and retrieves document embeddings via **ChromaDB**.
- **LLM-based Response Generation**: Uses **Zephyr-7B** model to answer queries.

## Installation
```bash
pip install langchain sentence-transformers chromadb
pip install -U langchain-community
pip install pypdf
```

## Setup
1. **Authenticate Hugging Face API**
```python
import os
from getpass import getpass

hf_token = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
```

2. **Load PDFs**
```python
from langchain.document_loaders import PyPDFDirectoryLoader
pdf_directory = "your/pdf/directory/path"  # Change this to your actual path
loader = PyPDFDirectoryLoader(pdf_directory)
data = loader.load()
```

3. **Text Splitting**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
text = text_splitter.split_documents(data)
```

4. **Create Embeddings and Store in ChromaDB**
```python
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma.from_documents(text, embeddings)
```

5. **Load Pretrained LLM Model**
```python
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

model = HuggingFaceHub(repo_id='HuggingFaceH4/zephyr-7b-alpha',
                        model_kwargs={'temperature': 0.5, 'max_new_tokens': 1024, 'max_length': 64})
retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 4})
qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")
```

## Query the Chatbot
```python
response = qa.invoke({"query": 'name all the diseases'})
print(response["result"])
```

## Expected Output
The chatbot will return a **medically relevant** response based on the retrieved documents.

## Future Improvements
- Integrate **custom prompts** for more precise responses.
- Deploy as a **Streamlit web app** for an interactive UI.
- Expand dataset with more medical literature.

## License
This project is open-source and available for further development and customization.

