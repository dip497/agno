import pytest

from agno.models.openai import OpenAIChat
from agno.team import Team


@pytest.fixture
def team(shared_db):
    """Create a route team with db and memory for testing."""

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny."

    return Team(
        model=OpenAIChat(id="gpt-5-mini"),
        members=[],
        tools=[get_weather],
        db=shared_db,
        instructions="Route a single question to the travel agent. Don't make multiple requests.",
        add_history_to_context=True,
    )


def test_history(team):
    response = team.run("What is the weather in Tokyo?")
    assert len(response.messages) == 5, "Expected system message, user message, assistant messages, and tool message"

    response = team.run("what was my first question? Say it verbatim.")
    assert "What is the weather in Tokyo?" in response.content
    assert response.messages is not None
    assert len(response.messages) == 7
    assert response.messages[0].role == "system"
    assert response.messages[1].role == "user"
    assert response.messages[1].content == "What is the weather in Tokyo?"
    assert response.messages[1].from_history is True
    assert response.messages[2].role == "assistant"
    assert response.messages[2].from_history is True
    assert response.messages[3].role == "tool"
    assert response.messages[3].from_history is True
    assert response.messages[4].role == "assistant"
    assert response.messages[4].from_history is True
    assert response.messages[5].role == "user"
    assert response.messages[5].from_history is False
    assert response.messages[6].role == "assistant"
    assert response.messages[6].from_history is False
