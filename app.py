import streamlit as st
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(
    page_title="Company Policy RAG",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Company Policy RAG")
st.write("Upload a company policy document and ask questions about it.")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "rag_database"


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name="company_policy",
        metadata={"hnsw:space": "cosine"}
    )

    return client, collection


client, collection = get_collection()


def clear_existing_documents():
    try:
        data = collection.get()

        if data["ids"]:
            collection.delete(ids=data["ids"])

        return True

    except Exception as e:
        st.error(f"Error clearing database: {e}")
        return False


def create_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


def process_document(text):
    chunks = create_chunks(text)

    if not chunks:
        return False

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return True


with st.sidebar:
    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a text file",
        type=["txt"]
    )

    if st.button("Clear Database"):
        if clear_existing_documents():
            st.success("Database cleared successfully.")
            st.rerun()


if uploaded_file is not None:

    try:
        document_text = uploaded_file.read().decode("utf-8")

        if st.button("Process Document"):

            with st.spinner("Processing document..."):

                clear_existing_documents()
                success = process_document(document_text)

            if success:
                st.success("Document processed successfully!")
                st.info(
                    f"Stored {collection.count()} chunks in ChromaDB."
                )
            else:
                st.error("Could not process the document.")

    except Exception as e:
        st.error(f"Error reading document: {e}")


st.divider()

st.header("❓ Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is the leave policy?"
)


def search_documents(question):

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results


if st.button("Ask AI"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning("Please upload and process a document first.")

    else:

        with st.spinner("Searching document..."):

            try:

                results = search_documents(question)

                documents = results.get("documents", [[]])[0]
                distances = results.get("distances", [[]])[0]

                if not documents:

                    st.warning(
                        "No relevant information was found."
                    )

                else:

                    context_parts = []

                    for i, document in enumerate(documents):

                        context_parts.append(
                            f"Document Chunk {i + 1}:\n{document}"
                        )

                    context = "\n\n".join(context_parts)

                    prompt = f"""
You are a precise company policy assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not present in the context, clearly say:
"I could not find this information in the provided document."

Do not invent information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

                    with st.spinner("Generating answer..."):

                        try:

                            client_ai = genai.Client(
                                api_key=st.secrets["GEMINI_API_KEY"]
                            )

                            response = client_ai.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=prompt
                            )

                            answer = response.text

                            st.success("Answer")
                            st.write(answer)

                        except Exception as e:

                            st.error(
                                f"Error generating answer: {e}"
                            )

                    with st.expander("📄 Retrieved Document Chunks"):

                        for i, document in enumerate(documents):

                            st.write(f"### Chunk {i + 1}")
                            st.write(document)

                            if i < len(distances):
                                st.caption(
                                    f"Distance: {distances[i]:.4f}"
                                )

            except Exception as e:

                st.error(
                    f"Error searching documents: {e}"
                )


with st.sidebar:

    st.divider()

    st.write("### Database Information")

    st.write(
        f"Stored chunks: **{collection.count()}**"
    )