import openai
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
# from langchain_community.vectorstores import Pinecone
from langchain_pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Set up your OpenAI and Pinecone API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("INDEX_NAME")

# Initialize Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Initialize LangChain components
vector_store = pc(index, embeddings)
model = ChatOpenAI(
    temperature=0.9, openai_api_key=openai.api_key, model_name="gpt-3.5-turbo"
)

# Initialize the RetrievalQA chain
chain = RetrievalQA.from_llm(
    llm=model,
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
)


def chat_with_vectors(prompt):
    response = chain({"query": prompt})
    response_text = response["result"]
    source_documents = response.get("source_documents", [])
    return response_text, source_documents


def main():
    print("Welcome to the GPT-3.5 Chatbot with Vectors! Type 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response_text, sources = chat_with_vectors(user_input)
        print(f"GPT-3.5: {response_text}")
        if sources:
            print("Sources:")
            for doc in sources:
                print(f"- {doc.metadata.get('source', 'Unknown Source')}")


if __name__ == "__main__":
    main()
