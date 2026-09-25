import sys
import os

import streamlit as st


project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(
    0,
    project_root
)


from src.rag.rag_service import RAGService


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Intelligence",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# Application title
# --------------------------------------------------

st.title(
    "📄 AI Document Intelligence & RAG Assistant"
)

st.write(
    "Ask questions about the loaded document "
    "and inspect the sources retrieved by the RAG pipeline."
)


# --------------------------------------------------
# Initialize conversation history
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Load RAG service
# --------------------------------------------------

@st.cache_resource
def load_rag_service():

    return RAGService()


with st.spinner(
    "Loading embedding and reranking models..."
):

    rag_service = load_rag_service()


# --------------------------------------------------
# Document information
# --------------------------------------------------

st.info(
    "📚 Current document: Metformin drug label "
    "(DailyMed/NLM)"
)


# --------------------------------------------------
# Display previous conversation
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# User question
# --------------------------------------------------

query = st.chat_input(
    "Ask a question about the document..."
)


# --------------------------------------------------
# Process question
# --------------------------------------------------

if query:

    # Store user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # Display user message

    with st.chat_message("user"):

        st.markdown(query)


    # Generate answer

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving relevant information..."
        ):

            result = rag_service.answer_query(
                query
            )


        if result["answer"]:

            st.markdown(
                result["answer"]
            )

            # Store assistant response

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"]
                }
            )


        elif result["error"]:

            error_message = (
                "Answer generation was not completed: "
                f"{result['error']}"
            )

            st.warning(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )


    # --------------------------------------------------
    # Retrieved sources for current question
    # --------------------------------------------------

    st.subheader(
        "📚 Retrieved Sources"
    )

    for index, chunk in enumerate(
        result["retrieved_chunks"],
        start=1
    ):

        with st.expander(
            f"Source {index} — "
            f"Page {chunk['page_number']}"
        ):

            st.write(
                f"**Document:** "
                f"{chunk['document']}"
            )

            st.write(
                f"**Page:** "
                f"{chunk['page_number']}"
            )

            st.write(
                f"**Reranker score:** "
                f"{chunk['score']:.4f}"
            )

            st.write(
                f"**Semantic score:** "
                f"{chunk['semantic_score']:.4f}"
            )

            st.write(
                "**Retrieved content:**"
            )

            st.write(
                chunk["text"]
            )


    # --------------------------------------------------
    # RAG pipeline details
    # --------------------------------------------------

    with st.expander(
        "🔍 View RAG Pipeline Details"
    ):

        st.markdown(
            """
            **RAG pipeline used for this answer**

            1. PDF document
            2. Text extraction
            3. Text cleaning and chunking
            4. Sentence-transformer embeddings
            5. Semantic similarity retrieval
            6. CrossEncoder reranking
            7. Top relevant chunks
            8. Grounded prompt construction
            9. Gemini LLM generation
            10. Source/page citations
            """
        )


    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------

    with st.expander(
        "🧠 View Grounded LLM Prompt"
    ):

        st.code(
            result["prompt"],
            language="text"
        )