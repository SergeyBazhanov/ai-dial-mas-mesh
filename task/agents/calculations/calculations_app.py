import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agents.calculations.calculations_agent import CalculationsAgent
from task.agents.calculations.tools.simple_calculator_tool import SimpleCalculatorTool
from task.tools.base_tool import BaseTool
from task.agents.calculations.tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool
from task.tools.deployment.content_management_agent_tool import ContentManagementAgentTool
from task.tools.deployment.web_search_agent_tool import WebSearchAgentTool
from task.utils.constants import DIAL_ENDPOINT, DEPLOYMENT_NAME

_PY_INTERPRETER_MCP_URL = os.getenv('PY_INTERPRETER_MCP_URL', "http://localhost:8050/mcp")


class CalculationsApplication(ChatCompletion):

    async def chat_completion(self, request: Request, response: Response) -> None:
        choice = response.create_single_choice()

        py_interpreter = await PythonCodeInterpreterTool.create(
            mcp_url=_PY_INTERPRETER_MCP_URL,
            tool_name="execute_python",
            dial_endpoint=DIAL_ENDPOINT,
        )

        tools: list[BaseTool] = [
            SimpleCalculatorTool(),
            py_interpreter,
            ContentManagementAgentTool(endpoint=DIAL_ENDPOINT),
            WebSearchAgentTool(endpoint=DIAL_ENDPOINT),
        ]

        agent = CalculationsAgent(
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
app.add_chat_completion("calculations-agent", CalculationsApplication())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
