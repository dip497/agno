"""
Example demonstrating how to use guardrails with an Agno Agent.

The AgentOS UI will show an error when the guardrail is triggered.
"""

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.guardrails import PromptInjectionGuardrail
from agno.models.openai import OpenAIChat
from agno.os import AgentOS

# Setup the database
db = PostgresDb(id="basic-db", db_url="postgresql+psycopg://ai:ai@localhost:5532/ai")

# Setup basic agents, teams and workflows
chat_agent = Agent(
    name="Chat Agent",
    model=OpenAIChat(id="gpt-5-mini"),
    pre_hooks=[PromptInjectionGuardrail()],
    instructions=[
        "You are a helpful assistant that can answer questions and help with tasks.",
        "Always answer in a friendly and helpful tone.",
        "Never be rude or offensive.",
    ],
    db=db,
    enable_user_memories=True,
    add_history_to_context=True,
    num_history_runs=3,
    add_datetime_to_context=True,
    markdown=True,
)

# Setup our AgentOS app
agent_os = AgentOS(
    description="Example app for chat agent with guardrails",
    agents=[chat_agent],
)
app = agent_os.get_app()


if __name__ == "__main__":
    agent_os.serve(app="guardrails_demo:app", reload=True)
