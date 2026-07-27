from app.agents.registry import AGENTS


class AgentExecutor:

    def __init__(self):
        self.results = {}

    def execute(self, plan):

        self.results = {}

        for step in plan:

            agent_name = step["agent"]
            task = step["task"]

            if agent_name not in AGENTS:
                print(f"Unknown agent: {agent_name}")
                continue

            print(f"Running {agent_name}...")

            agent = AGENTS[agent_name]

            result = agent(task)

            self.results[agent_name] = result

        return self.results