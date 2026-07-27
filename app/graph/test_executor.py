from app.graph.executor import AgentExecutor

plan = [
    {
        "agent": "web",
        "task": "Redis blogs"
    },
    {
        "agent": "github",
        "task": "Redis repositories"
    },
    {
        "agent": "papers",
        "task": "Redis papers"
    },
    {
        "agent": "memory",
        "task": "Redis notes"
    }
]

executor = AgentExecutor()

results = executor.execute(plan)

print(results)