from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""


from langchain.chat_models import init_chat_model
model = init_chat_model(model="meta-llama/llama-4-maverick-17b-128e-instruct",
                        model_provider="groq",
                        temperature = 0,
                        max_retries=3,
                        timeout=60,
                        max_tokens=1000
                        )

from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=system_prompt
)