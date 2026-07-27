import streamlit as st

from app.graph.workflow import graph


st.set_page_config(
    page_title="AI Research Platform",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Platform")

st.caption(
    "Multi-agent research across the web, GitHub, academic papers, "
    "and long-term memory."
)

query = st.text_area(
    "Research question",
    placeholder=(
        "Example: Research LangGraph persistence and find "
        "documentation, GitHub implementations and academic papers."
    ),
    height=120,
)

if st.button("Research", type="primary", disabled=not query.strip()):

    try:
        with st.spinner("Research agents are working..."):

            result = graph.invoke({
                "query": query.strip()
            })

        report = result.get("report")

        if report:
            st.markdown(report)
        else:
            st.warning("Research completed, but no report was generated.")

    except Exception as exc:
        st.error("Research failed.")
        st.exception(exc)