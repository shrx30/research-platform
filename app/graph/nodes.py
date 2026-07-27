from app.graph.state import ResearchState
from app.schemas.planner import ExecutionPlan
from app.llm.models import planner_llm, research_llm

from app.agents.web import run as web_run
from app.agents.github import run as github_run
from app.agents.papers import run as papers_run
from app.agents.memory import run as memory_run
from app.agents.memory_writer import write_memories


# =========================================================
# PLANNER
# =========================================================

def planner_node(state: ResearchState):

    prompt = f"""
You are the Planner Agent for a multi-agent AI research system.

Your job is ONLY to decide which specialized agents are needed
and create an execution plan.

Available agents:

web
Use for:
- current information
- documentation
- tutorials
- news
- websites
- recent developments

github
Use for:
- repositories
- source code
- implementations
- README files
- open-source projects
- code examples

papers
Use for:
- academic papers
- research publications
- scientific literature

memory
Use for:
- information previously stored in long-term memory


For every selected agent generate:

agent:
The specialized agent to execute.

task:
A natural-language description of what that agent should accomplish.

query:
A concise search query optimized for that agent's retrieval system.


IMPORTANT QUERY RULES:

1. Generate the query dynamically from the user's request.

2. Do NOT put instructions such as:
   "Search for..."
   "Find..."
   "Look for..."
   inside query.

3. query should contain search terms, not instructions.

4. Preserve important:
   - technology names
   - project names
   - people
   - organizations
   - technical terminology

5. GitHub queries should focus on repository/topic keywords.

6. Papers queries should focus on academic concepts and terminology.

7. Web queries can be more descriptive and should preserve
   recency requirements when relevant.

8. Memory queries should represent the knowledge that needs
   to be retrieved.

9. Use the minimum number of agents necessary.

10. Do not use memory unless previously stored information
    could actually help answer the request.

11. Do not invent agents.

12. Do not answer the user's question yourself.


Example:

User request:
Research LangGraph implementations and academic research.

Possible plan:

github
task: Find useful open-source LangGraph implementations and examples.
query: LangGraph

papers
task: Find academic research involving LangGraph.
query: LangGraph


Another example:

User request:
Research recent multi-agent memory architectures and find implementations.

Possible plan:

web
task: Find recent developments in multi-agent memory architectures.
query: multi-agent LLM memory architectures

github
task: Find open-source implementations of multi-agent memory systems.
query: multi-agent memory LLM agents

papers
task: Find academic research about memory architectures in multi-agent LLM systems.
query: multi-agent LLM memory architecture


USER REQUEST:

{state["query"]}
"""

    plan: ExecutionPlan = planner_llm.invoke(prompt)

    print("\n========== PLAN ==========")

    for step in plan.steps:
        print(f"\n{step.agent}")
        print(f"  Task:  {step.task}")
        print(f"  Query: {step.query}")

    print("\n==========================\n")

    return {
        "plan": plan.steps
    }


# =========================================================
# PLAN STEP HELPER
# =========================================================

def _get_step(state: ResearchState, agent_name: str):

    for step in state["plan"]:
        if step.agent == agent_name:
            return step

    raise ValueError(
        f"No plan step found for agent '{agent_name}'"
    )


# =========================================================
# WEB AGENT
# =========================================================

def web_agent(state: ResearchState):

    step = _get_step(state, "web")

    print(f"[WEB] Task:  {step.task}")
    print(f"[WEB] Query: {step.query}")

    result = web_run(step.query)

    return {
        "web_context": result
    }


# =========================================================
# GITHUB AGENT
# =========================================================

def github_agent(state: ResearchState):

    step = _get_step(state, "github")

    print(f"[GITHUB] Task:  {step.task}")
    print(f"[GITHUB] Query: {step.query}")

    result = github_run(step.query)

    return {
        "github_context": result
    }


# =========================================================
# PAPERS AGENT
# =========================================================

def paper_agent(state: ResearchState):

    step = _get_step(state, "papers")

    print(f"[PAPERS] Task:  {step.task}")
    print(f"[PAPERS] Query: {step.query}")

    result = papers_run(
    query=step.query,
    user_query=state["query"],
     ) 

    return {
        "paper_context": result
    }


# =========================================================
# MEMORY AGENT
# =========================================================

def memory_agent(state: ResearchState):

    query = state["query"]

    print(f"[MEMORY] Searching: {query}")

    result = memory_run(query)

    print(
        "[MEMORY] Result:",
        "found" if result != "No memory context." else "none"
    )

    return {
        "memory_context": result
    }

def memory_write_node(state: ResearchState):
    result = state.get("research_result")

    if result is None:
        print("[MEMORY] No research result. Skipping memory write.")
        return {}

    try:
        count = write_memories(
            user_query=state["query"],
            research_result=result,
        )

        print(f"[MEMORY] Stored {count} memories")

    except Exception as exc:
        print(f"[MEMORY] Write failed: {exc}")
        # Do not kill an otherwise successful research run.

    return {}
# =========================================================
# MERGE
# =========================================================

def merge_node(state: ResearchState):

    web_context = state.get("web_context", "")
    github_context = state.get("github_context", "")
    paper_context = state.get("paper_context", "")
    memory_context = state.get("memory_context", "")

    # Temporary debugging
    print("\n========== MERGE DEBUG ==========")

    print("\n--- WEB ---")
    print(web_context[:1500] if web_context else "No web context.")

    print("\n--- GITHUB ---")
    print(
        github_context[:1500]
        if github_context
        else "No GitHub context."
    )

    print("\n--- PAPERS ---")
    print(
        paper_context[:2000]
        if paper_context
        else "No paper context."
    )

    print("\n--- MEMORY ---")
    print(
        memory_context[:1000]
        if memory_context
        else "No memory context."
    )

    print("\n=================================\n")

    merged_context = f"""
=========================
WEB RESULTS
=========================

{web_context}


=========================
GITHUB RESULTS
=========================

{github_context}


=========================
ACADEMIC PAPERS
=========================

{paper_context}


=========================
LONG TERM MEMORY
=========================

{memory_context}
"""

    return {
        "merged_context": merged_context
    }


# =========================================================
# RESEARCH
# =========================================================

def research_node(state: ResearchState):

    prompt = f"""
You are an expert AI Research Assistant.

Synthesize the evidence collected by specialized research agents.


=========================
USER QUESTION
=========================

{state["query"]}


=========================
RESEARCH CONTEXT
=========================

{state["merged_context"]}


=========================
INSTRUCTIONS
=========================

1. Answer the user's actual question.

2. Read all available source categories.

3. Use all relevant evidence.

4. Prefer academic papers for scientific and research claims.

5. Use GitHub results for:
   - implementations
   - repositories
   - code
   - libraries
   - examples

6. Use web results for:
   - current information
   - documentation
   - tutorials
   - announcements

7. Use memory only when useful information actually exists.

8. Distinguish between:
   - no results returned
   - results returned but irrelevant
   - relevant evidence found

9. Never invent:
   - facts
   - URLs
   - repositories
   - papers
   - authors
   - companies
   - statistics
   - citations

10. Do not claim GitHub results are missing if relevant
    GitHub results exist in the supplied context.

11. Do not claim academic research is missing if relevant
    papers exist in the supplied context.

12. If retrieved evidence is irrelevant to the user's question,
    do not use it merely because it was retrieved.

13. If sources conflict, explain the disagreement.

14. sources_used must contain only sources actually present
    in the supplied context.

15. Explicitly identify genuinely missing information.

16. Set confidence based on evidence quality, relevance,
    coverage, and agreement.

17. Return a complete structured ResearchResult.
"""

    result = research_llm.invoke(prompt)

    return {
        "research_result": result
    }

def research_node(state: ResearchState):
    prompt = f"""
    # your existing prompt
    """

    result = research_llm.invoke(prompt)

    print("\n========== RESEARCH LLM DEBUG ==========")
    print("RESULT:", result)
    print("TYPE:", type(result))
    print("========================================\n")

    if result is None:
        raise RuntimeError(
            "research_llm returned None. "
            "Structured output parsing/model compatibility likely failed."
        )

    return {
        "research_result": result
    }


# =========================================================
# REPORT
# =========================================================

def report_node(state: ResearchState):

    result = state["research_result"]

    report = f"""# 📄 Research Report

## Executive Summary

{result.summary}

---

## Key Findings

"""

    if result.key_findings:
        for finding in result.key_findings:
            report += f"- {finding}\n"
    else:
        report += "No key findings were generated.\n"

    report += "\n---\n\n## Sources Used\n\n"

    if result.sources_used:
        for source in result.sources_used:
            report += f"- {source}\n"
    else:
        report += "No sources were recorded.\n"

    report += "\n---\n\n## Missing Information\n\n"

    if result.missing_information:
        for item in result.missing_information:
            report += f"- {item}\n"
    else:
        report += "None\n"

    report += f"""
---

## Confidence

{result.confidence}
"""

    return {
        "report": report
    }