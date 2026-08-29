import time
import traceback

import streamlit as st

from app.graph.workflow import graph
from app.voice.asr import transcribe_audio
from app.voice.tts import text_to_speech


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Multi-Agent Research Platform",
    page_icon="🧠",
    layout="wide",
)


# =========================================================
# SESSION STATE
# =========================================================

if "query" not in st.session_state:
    st.session_state.query = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "last_latency" not in st.session_state:
    st.session_state.last_latency = None

if "audio_response" not in st.session_state:
    st.session_state.audio_response = None


# =========================================================
# HEADER
# =========================================================

st.title("🧠 Multi-Agent Research Platform")

st.markdown(
    """
Research using specialized AI agents for:

- 🌐 Web research
- 🐙 GitHub research
- 📚 Academic papers
- 🧠 Long-term memory
- 🎙️ Voice interaction
- 🔄 Multi-agent orchestration
"""
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Research Harness")

    st.markdown(
        """
### Pipeline

`Query`

↓

`Planner`

↓

`Dynamic Tool Routing`

↓

`Parallel Research Agents`

↓

`Evidence Merge`

↓

`Research Synthesis`

↓

`Report + Memory`

### Agents

- Web
- GitHub
- Papers
- Memory

### Engineering

- Structured outputs
- Source validation
- Evidence budgeting
- Semantic memory
- Latency tracking
- Guardrails
"""
    )

    st.divider()

    if st.session_state.last_latency is not None:

        st.metric(
            "Last Run",
            f"{st.session_state.last_latency:.2f}s",
        )


# =========================================================
# VOICE INPUT
# =========================================================

st.subheader("🎙️ Voice Research")

audio_file = st.audio_input(
    "Ask your research question by voice"
)

if audio_file is not None:

    try:

        with st.spinner("Transcribing..."):

            transcript = transcribe_audio(
                audio_file
            )

        if transcript:

            st.session_state.query = transcript

            st.success(
                "Voice transcribed successfully."
            )

            st.write(
                f"**You said:** {transcript}"
            )

    except Exception as exc:

        st.error(
            f"Voice transcription failed: {exc}"
        )


# =========================================================
# TEXT QUERY
# =========================================================

st.subheader("🔎 Research Query")

query = st.text_area(
    "What do you want to research?",
    value=st.session_state.query,
    height=120,
    placeholder=(
        "Example: Research recent developments "
        "in multi-agent memory systems."
    ),
)

st.session_state.query = query


# =========================================================
# EXAMPLE QUERIES
# =========================================================

st.caption("Try an example:")

col1, col2, col3 = st.columns(3)

with col1:

    if st.button(
        "Multi-agent memory",
        use_container_width=True,
    ):

        st.session_state.query = (
            "Research recent developments "
            "in multi-agent memory systems."
        )

        st.rerun()


with col2:

    if st.button(
        "RAG frameworks",
        use_container_width=True,
    ):

        st.session_state.query = (
            "Find open-source RAG frameworks "
            "and explain their current features."
        )

        st.rerun()


with col3:

    if st.button(
        "Vision Transformers",
        use_container_width=True,
    ):

        st.session_state.query = (
            "Find GitHub implementations "
            "of Vision Transformers."
        )

        st.rerun()


# =========================================================
# RUN RESEARCH
# =========================================================

st.divider()

run_research = st.button(
    "🚀 Run Research",
    type="primary",
    use_container_width=True,
)


if run_research:

    query = st.session_state.query.strip()

    if not query:

        st.warning(
            "Please enter a research question."
        )

        st.stop()

    # -----------------------------------------------------
    # RUN GRAPH
    # -----------------------------------------------------

    start_time = time.perf_counter()

    try:

        with st.spinner(
            "Research agents are working..."
        ):

            result = graph.invoke(
                {
                    "query": query,
                }
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        st.session_state.result = result
        st.session_state.last_latency = elapsed

        st.success(
            f"Research completed in {elapsed:.2f}s"
        )

    except Exception as exc:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        st.session_state.last_latency = elapsed

        st.error(
            "Research pipeline failed."
        )

        st.code(
            str(exc)
        )

        with st.expander(
            "Technical traceback"
        ):

            st.code(
                traceback.format_exc()
            )

        st.stop()


# =========================================================
# DISPLAY RESULT
# =========================================================

result = st.session_state.result


if result is not None:

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================

    research_result = result.get(
        "research_result"
    )

    memories_written = result.get(
        "memories_written",
        0,
    )

    plan = result.get(
        "plan",
        [],
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Latency",
            (
                f"{st.session_state.last_latency:.2f}s"
                if st.session_state.last_latency
                else "N/A"
            ),
        )

    with col2:

        st.metric(
            "Agents Selected",
            len(plan),
        )

    with col3:

        if research_result is not None:

            findings = getattr(
                research_result,
                "key_findings",
                [],
            )

            st.metric(
                "Key Findings",
                len(findings),
            )

        else:

            st.metric(
                "Key Findings",
                0,
            )

    with col4:

        st.metric(
            "Memories Written",
            memories_written,
        )


    # =====================================================
    # FINAL REPORT
    # =====================================================

    report = result.get(
        "report"
    )

    st.subheader(
        "📄 Research Report"
    )

    if report:

        st.markdown(
            report
        )

    else:

        st.warning(
            "Research completed, but no report was generated."
        )

        # -------------------------------------------------
        # Fallback display
        # -------------------------------------------------

        if research_result is not None:

            st.markdown(
                "### Summary"
            )

            st.write(
                getattr(
                    research_result,
                    "summary",
                    "",
                )
            )

            findings = getattr(
                research_result,
                "key_findings",
                [],
            )

            if findings:

                st.markdown(
                    "### Key Findings"
                )

                for finding in findings:

                    st.markdown(
                        f"- {finding}"
                    )


    # =====================================================
    # STRUCTURED RESEARCH RESULT
    # =====================================================

    if research_result is not None:

        with st.expander(
            "🔬 Structured Research Result"
        ):

            try:

                if hasattr(
                    research_result,
                    "model_dump",
                ):

                    st.json(
                        research_result.model_dump()
                    )

                else:

                    st.json(
                        research_result.dict()
                    )

            except Exception:

                st.write(
                    research_result
                )


    # =====================================================
    # SOURCES
    # =====================================================

    if research_result is not None:

        sources = getattr(
            research_result,
            "sources_used",
            [],
        )

        if sources:

            st.subheader(
                "🔗 Sources"
            )

            for source in sources:

                st.markdown(
                    f"- {source}"
                )


    # =====================================================
    # MISSING INFORMATION
    # =====================================================

    if research_result is not None:

        missing = getattr(
            research_result,
            "missing_information",
            [],
        )

        if missing:

            with st.expander(
                "⚠️ Missing Information"
            ):

                for item in missing:

                    st.markdown(
                        f"- {item}"
                    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    if research_result is not None:

        confidence = getattr(
            research_result,
            "confidence",
            None,
        )

        if confidence:

            st.subheader(
                "Confidence"
            )

            st.info(
                str(confidence)
            )


    # =====================================================
    # AGENT ROUTING
    # =====================================================

    if plan:

        with st.expander(
            "🧭 Agent Routing"
        ):

            for item in plan:

                if isinstance(
                    item,
                    dict,
                ):

                    agent = item.get(
                        "agent",
                        item.get(
                            "name",
                            "unknown",
                        ),
                    )

                    search_query = item.get(
                        "query",
                        "",
                    )

                    st.markdown(
                        f"**{agent}**"
                    )

                    if search_query:

                        st.code(
                            search_query
                        )

                else:

                    st.write(
                        item
                    )


    # =====================================================
    # RAW AGENT EVIDENCE
    # =====================================================

    with st.expander(
        "🧪 Raw Agent Evidence"
    ):

        web_results = result.get(
            "web_results",
            "",
        )

        github_results = result.get(
            "github_results",
            "",
        )

        paper_results = result.get(
            "paper_results",
            "",
        )

        memory_results = result.get(
            "memory_results",
            "",
        )

        if web_results:

            st.markdown(
                "### 🌐 Web"
            )

            st.text(
                web_results
            )

        if github_results:

            st.markdown(
                "### 🐙 GitHub"
            )

            st.text(
                github_results
            )

        if paper_results:

            st.markdown(
                "### 📚 Papers"
            )

            st.text(
                paper_results
            )

        if memory_results:

            st.markdown(
                "### 🧠 Memory"
            )

            st.text(
                memory_results
            )


    # =====================================================
    # VOICE RESPONSE
    # =====================================================

    st.divider()

    st.subheader(
        "🔊 Voice Response"
    )

    if report:

        generate_voice = st.button(
            "Generate Audio Response",
            use_container_width=True,
        )

        if generate_voice:

            try:

                with st.spinner(
                    "Generating voice response..."
                ):

                    audio = text_to_speech(
                        report
                    )

                if audio:

                    st.session_state.audio_response = audio

                    st.audio(
                        audio,
                        format="audio/wav",
                    )

            except Exception as exc:

                st.error(
                    f"Voice generation failed: {exc}"
                )

    if (
        st.session_state.audio_response
        and not run_research
    ):

        st.audio(
            st.session_state.audio_response,
            format="audio/wav",
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Multi-Agent Research Platform • "
    "LangGraph • Dynamic Routing • "
    "Evidence Synthesis • Persistent Memory • Voice"
)
