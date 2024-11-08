import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from tools import cosine_similarity

history = []

# Load a model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define disease-related sample questions
disease_related_samples = [
    "What are the symptoms of cardiovascular disease?",
    "How can I prevent diabetes?",
    "What treatments are available for cancer?"
]

# Encode the disease-related questions
disease_embeddings = model.encode(disease_related_samples, convert_to_tensor=True)


def is_disease_related(query):
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Initialize a list to store similarity scores
    similarity_scores = []

    # Calculate cosine similarity for each disease embedding
    for sample_embedding in disease_embeddings:
        similarity_scores.append(cosine_similarity(query_embedding, sample_embedding))

    # Get the maximum similarity score
    max_similarity = max(similarity_scores)

    # Check if the max similarity exceeds the threshold
    return max_similarity >= 0.5

# Define the generic response
GENERIC_RESPONSE = "I'm here to help with healthcare-related queries. Could you please ask about a specific health topic?"

# Configure the embedding model
def get_embedder():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Load and split documents
def load_and_split_documents(file_path, chunk_size=1000, chunk_overlap=200):
    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

# Create vector store
def create_vector_store(docs, embedder):
    return FAISS.from_documents(docs, embedder)

# Format documents as string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Process directory
def process_directory(data_dir):
    embedder = get_embedder()
    all_splits = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            print(f"Processing file: {filepath}")
            splits = load_and_split_documents(filepath)
            all_splits.extend(splits)
    vectorstore = create_vector_store(all_splits, embedder)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create RAG chain
def create_rag_chain(data_dir, model_name="llama3.2"):
    retriever = process_directory(data_dir)
    llm = OllamaLLM(model=model_name)

    prompt = """
    You are a doctor that helps to answer questions based on the context provided. The context is information from documents related to healthcare topics.
    Answer the following question based on the context below. If you do not know the answer, simply say 'I don't know'.

    Context:
    {context}

    Question: {question}
    Answer:
    """

    def retrieve_and_format(query):
        retrieved_docs = retriever.invoke(query)
        return format_docs(retrieved_docs)

    def rag_chain(query):
        if not is_disease_related(query):
            answer = llm.invoke(query)
            return answer
        context = retrieve_and_format(query)
        prompt_with_context = prompt.format(context=context, question=query)
        answer = llm.invoke(prompt_with_context)
        return answer

    return rag_chain

# Function to ask a question
def ask_question(chain, question):
    return chain(question)


# Use this in the Gradio chatbot function
def handle_question(question, _):
    # Combine the last few messages in history with the current question for context
    context = ""
    if history:
        # Use the last few turns in the history to create a context
        context = "\n".join([f"User: {msg[0]}\nBot: {msg[1]}" for msg in history[-3:]])

    # Append the current question to the context
    context += f"\nUser: {question}\n"

    # Generate a response based on the question and the context
    response = ask_question(rag_chain, context)

    # Append the latest interaction to the history
    history.append((question, response))

    return response


if __name__ == "__main__":
    data_dir = 'C:/Users/7420/Desktop/Code/Python/RAG_Healthcare/data'
    rag_chain = create_rag_chain(data_dir)

    # Launch the Gradio Chat Interface
    gr.ChatInterface(
        fn=handle_question,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me a question related to healthcare and diseases", container=False),
        title="Healthcare Chatbot",
        examples=[
            "What are different kinds of diseases?",
            "How to prevent heart disease?",
            "Symptoms of diabetes"
        ],
        cache_examples=False  # Disable caching if you don't need cached examples
    ).launch(share=True)