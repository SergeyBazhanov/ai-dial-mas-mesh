import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.web_search.web_search_agent import WebSearchAgent
from task.tools.base_tool import BaseTool
from task.tools.deployment.calculations_agent_tool import CalculationsAgentTool
from task.tools.deployment.content_management_agent_tool import ContentManagementAgentTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME

_DDG_MCP_URL = os.getenv('DDG_MCP_URL', "http://localhost:8051/mcp")


class WebSearchApplication(ChatCompletion):

    async def chat_completion(self, request: Request, response: Response) -> None:
        choice = response.create_single_choice()

        # Create MCP tools from DDG MCP server
        mcp_client = await MCPClient.create(_DDG_MCP_URL)
        mcp_tool_models = await mcp_client.get_tools()
        mcp_tools: list[BaseTool] = [
            MCPTool(client=mcp_client, mcp_tool_model=tool_model)
            for tool_model in mcp_tool_models
        ]

        tools: list[BaseTool] = [
            *mcp_tools,
            CalculationsAgentTool(endpoint=DIAL_ENDPOINT),
            ContentManagementAgentTool(endpoint=DIAL_ENDPOINT),
        ]

        agent = WebSearchAgent(
            endpoint=DIAL_ENDPOINT,
            tools=tools,
        )

        await agent.handle_request(
            deployment_name=DEPLOYMENT_NAME,
            choice=choice,
            request=request,
            response=response,
        )


app = DIALApp()
app.add_chat_completion("web-search-agent", WebSearchApplication())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5003)
