# Multi-Agent Research Platform

A multi-agent research assistant built with **LangGraph** that dynamically plans research tasks, searches across multiple sources, generates grounded reports, maintains long-term memory, and supports **voice-based interaction**.

The project focuses on dynamic agent routing, parallel retrieval, grounded synthesis, research memory, voice interaction, evaluation, and low-latency execution.

## Features

- Dynamic research planning and agent routing
- Web research
- GitHub repository search
- arXiv paper search
- Long-term research memory
- **Voice input for research queries**
- **Voice-based interaction with the research assistant**
- Parallel research agent execution
- Evidence aggregation
- Grounded research synthesis
- Source tracking
- Automatic research reports
- Planner evaluation
- Retrieval relevance evaluation
- Source validity evaluation
- LLM groundedness evaluation
- Latency monitoring

## Architecture

```text
                 Text / Voice Query
                        │
                        ▼
                  Voice Processing
                  (when required)
                        │
                        ▼
                     Planner
                        │
                  Dynamic Routing
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
       Web            GitHub          Papers
        │               │               │
        └───────────────┼───────────────┘
                        │
                  Optional Memory
                        │
                        ▼
                      Merge
                        │
                        ▼
                    Synthesis
                        │
                        ▼
                     Report
                        │
                        ▼
               Text / Voice Response
```

The planner selects only the research agents required for a query instead of executing every available agent.

For example:

```text
"Find GitHub implementations of Vision Transformers"

→ GitHub Agent
```

while:

```text
"Research recent developments in multi-agent memory systems"

→ Web Agent + Papers Agent
```

Selected agents execute in parallel before their evidence is merged and synthesized.

## Voice Interaction

The platform also supports voice interaction, allowing research queries to be initiated through speech rather than only typed input.

```text
User Speech
     ↓
Voice Input
     ↓
Research Query
     ↓
Multi-Agent Research Pipeline
     ↓
Generated Answer
     ↓
Voice / Text Response
```

This makes the research system usable as a conversational research assistant in addition to a traditional text-based research tool.

## Tech Stack

- Python
- LangGraph
- LangChain
- NVIDIA NIM
- Pydantic
- arXiv API
- GitHub API
- Vector-based long-term memory
- Speech / Voice integration