import streamlit as st
from rag_utils import build_knowledge_base, answer_question

st.set_page_config(page_title="Document RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Document RAG Assistant")
st.write("Upload PDF or TXT files, then ask questions based on their content.")

def get_api_key():
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return ""

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if st.button("Process documents", type="primary", disabled=not uploads):
        with st.spinner("Reading documents and building search index..."):
            try:
                st.session_state["rag_data"] = build_knowledge_base(uploads)
                st.session_state["chat"] = []
                st.success(f"Ready! Created {len(st.session_state['rag_data']['chunks'])} chunks.")
            except Exception as exc:
                st.session_state.pop("rag_data", None)
                st.error(f"Could not process documents: {exc}")

    if st.button("Clear documents and chat"):
        st.session_state.pop("rag_data", None)
        st.session_state["chat"] = []
        st.rerun()

if "chat" not in st.session_state:
    st.session_state["chat"] = []

for item in st.session_state["chat"]:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item.get("sources"):
            with st.expander("Sources"):
                for source in item["sources"]:
                    st.markdown(f"**{source['name']} — chunk {source['chunk']}**")
                    st.caption(source["text"])

rag_data = st.session_state.get("rag_data")
if rag_data:
    st.success(f"Knowledge base ready: {len(rag_data['chunks'])} chunks")
else:
    st.info("Upload documents in the sidebar and click **Process documents** to begin.")

question = st.chat_input("Ask a question about your documents...", disabled=not rag_data)
if question:
    st.session_state["chat"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        api_key = get_api_key()
        if not api_key:
            response = "Groq API key is missing. Add GROQ_API_KEY in your Streamlit app's Secrets."
            st.error(response)
            st.session_state["chat"].append({"role": "assistant", "content": response})
        else:
            with st.spinner("Finding relevant text and generating an answer..."):
                try:
                    response, sources = answer_question(
                        question=question,
                        rag_data=rag_data,
                        api_key=api_key,
                        model=st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")
                    )
                    st.markdown(response)
                    if sources:
                        with st.expander("Sources used"):
                            for source in sources:
                                st.markdown(f"**{source['name']} — chunk {source['chunk']}**")
                                st.caption(source["text"])
                    st.session_state["chat"].append(
                        {"role": "assistant", "content": response, "sources": sources}
                    )
                except Exception as exc:
                    response = f"Unable to generate an answer: {exc}"
                    st.error(response)
                    st.session_state["chat"].append({"role": "assistant", "content": response})
