from app.agents.web import run as web

from app.agents.github import run as github

from app.agents.papers import run as papers

from app.agents.memory import run as memory

AGENTS = {
    "web": web,
    "github": github,
    "papers": papers,
    "memory": memory,
}