import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Session state initialization
# -----------------------------
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Utility: Extract YouTube ID
# -----------------------------
def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    if len(url) == 11:
        return url
    raise ValueError("Invalid YouTube URL or video ID")

# -----------------------------
# Build RAG chain
# -----------------------------
def build_chain(youtube_url: str):
    video_id = extract_video_id(youtube_url)

    # Fetch transcript
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    transcript = " ".join(snippet.text for snippet in transcript_list)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY using the provided video transcript context.
If the context is insufficient, reply with "I don't know".

Context:
{context}

Question: {question}
""",
        input_variables=['context', 'question']
    )

    # Format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    return parallel_chain | prompt | llm | StrOutputParser()

# -----------------------------
# Streamlit UI
# -----------------------------
#st.title("📺 YouTube Video Chatbot")

st.markdown(
    """
    <h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" 
             width="30" height="30" style="vertical-align:middle;"> 
        YouTube Video Chatbot
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Enter a YouTube URL and ask questions about its content.")

youtube_url = st.text_input("YouTube URL", key="youtube_url")

# Build chain button
if st.button("Load Video Transcript"):
    if youtube_url:
        with st.spinner("Fetching transcript and building chain..."):
            try:
                st.session_state.qa_chain = build_chain(youtube_url)
                st.success("Video loaded! Ask your questions below.")
                st.session_state.history = []  # Reset chat history
            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------
# Function to handle new questions
# -----------------------------
def ask_question():
    question = st.session_state.question_input
    if question and st.session_state.qa_chain:
        answer = st.session_state.qa_chain.invoke(question)
        st.session_state.history.append({"question": question, "answer": answer})
        st.session_state.question_input = ""  # Clear input

# -----------------------------
# Chat input at bottom
# -----------------------------
if st.session_state.qa_chain:
    st.text_input(
        "Ask a question about the video:",
        key="question_input",
        on_change=ask_question
    )

# -----------------------------
# Display chat history
# -----------------------------
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    st.markdown("---")
