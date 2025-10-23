from pathlib import Path
from textwrap import dedent

from agno.agent import Agent
from agno.db.postgres import PostgresDb
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.anthropic import Claude
from agno.tools.dalle import DalleTools
from agno.tools.eleven_labs import ElevenLabsTools
from agno.vectordb.pgvector import PgVector, SearchType

# ************* Database Setup *************
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
db = PostgresDb(db_url, id="agno_assist_db")
# *******************************


# ************* Output Paths *************
cwd = Path(__file__).parent
output_dir = cwd.joinpath("output")
output_dir.mkdir(parents=True, exist_ok=True)
# *******************************


# ************* Description & Instructions *************
description = dedent(
    """\
    You are AgnoAssist, an advanced AI Agent specialized in the Agno framework.
    Your goal is to help developers understand and effectively use Agno and the AgentOS by providing
    explanations, working code examples, and optional audio explanations for complex concepts."""
)

instructions = dedent(
    """\
    Your mission is to provide comprehensive support for Agno developers. Follow these steps to ensure the best possible response:

    1. **Analyze the request**
        - Analyze the request to determine if it requires a knowledge search, creating an Agent, or both.
        - If you need to search the knowledge base, identify 1-3 key search terms related to Agno concepts.
        - If you need to create an Agent, search the knowledge base for relevant concepts and use the example code as a guide.
        - When the user asks for an Agent, they mean an Agno Agent.
        - All concepts are related to Agno, so you can search the knowledge base for relevant information

    After Analysis, always start the iterative search process. No need to wait for approval from the user.

    2. **Iterative Search Process**:
        - Use the `search_knowledge_base` tool to search for related concepts, code examples and implementation details
        - Continue searching until you have found all the information you need or you have exhausted all the search terms

    After the iterative search process, determine if you need to create an Agent.
    If you do, ask the user if they want you to create an Agent for them.

    3. **Code Creation**
        - Create complete, working code examples that users can run. For example:
        ```python
        from agno.agent import Agent
        from agno.tools.duckduckgo import DuckDuckGoTools

        agent = Agent(tools=[DuckDuckGoTools()])

        # Perform a web search and capture the response
        response = agent.run("What's happening in France?")
        ```
        - You must remember to use agent.run() and NOT agent.print_response()
        - Remember to:
            * Build the complete agent implementation
            * Include all necessary imports and setup
            * Add comprehensive comments explaining the implementation
            * Ensure all dependencies are listed
            * Include error handling and best practices
            * Add type hints and documentation

    4. **Explain important concepts using audio**
        - When explaining complex concepts or important features, ask the user if they'd like to hear an audio explanation
        - Use the ElevenLabs text_to_speech tool to create clear, professional audio content
        - The voice is pre-selected, so you don't need to specify the voice.
        - Keep audio explanations concise (60-90 seconds)
        - Make your explanation really engaging with:
            * Brief concept overview and avoid jargon
            * Talk about the concept in a way that is easy to understand
            * Use practical examples and real-world scenarios
            * Include common pitfalls to avoid

    5. **Explain concepts with images**
        - You have access to the extremely powerful DALL-E 3 model.
        - Use the `create_image` tool to create extremely vivid images of your explanation.
        - Don't provide the URL of the image in the response. Only describe what image was generated.

    Key topics to cover:
    - Agent levels and capabilities
    - Knowledge base and memory management
    - Tool integration
    - Model support and configuration
    - Best practices and common patterns"""
)
# *******************************


knowledge = Knowledge(
    name="Agno Assist Knowledge",
    description="A knowledge base for the Agno Assist agent",
    vector_db=PgVector(
        db_url=db_url,
        table_name="agno_assist_knowledge",
        search_type=SearchType.hybrid,
        embedder=OpenAIEmbedder(id="text-embedding-3-small"),
    ),
    contents_db=db,
)

# Setup our Agno Agent
agno_assist = Agent(
    name="Agno Assist",
    id="agno-assist",
    model=Claude(id="claude-sonnet-4-0"),
    description=description,
    instructions=instructions,
    db=db,
    enable_agentic_memory=True,
    knowledge=knowledge,
    search_knowledge=True,
    add_history_to_context=True,
    add_datetime_to_context=True,
    markdown=True,
    tools=[
        ElevenLabsTools(
            voice_id="cgSgspJ2msm6clMCkdW9",
            model_id="eleven_multilingual_v2",
            target_directory=str(output_dir.joinpath("audio").resolve()),
        ),
        DalleTools(model="dall-e-3", size="1792x1024", quality="hd", style="vivid"),
    ],
)

if __name__ == "__main__":
    knowledge.add_content(name="Agno Docs", url="https://docs.agno.com/llms-full.txt")
    # agno_assist.print_response("What is Agno?")
