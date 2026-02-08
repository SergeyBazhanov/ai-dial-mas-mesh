import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.tools.base_tool import BaseTool
from task.tools.models import ToolCallParams
from task.utils.stage import StageProcessor


class BaseAgentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Parse prompt and propagate_history from tool call arguments
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt", "")
        propagate_history = arguments.get("propagate_history", False)

        # 2. Create AsyncDial client and call the agent with streaming
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview'
        )

        messages = self._prepare_messages(tool_call_params)

        chunks = await client.chat.completions.create(
            messages=messages,
            stream=True,
            deployment_name=self.deployment_name,
            extra_headers={"x-conversation-id": tool_call_params.conversation_id}
        )

        # 3. Prepare collection variables
        content = ''
        custom_content: CustomContent = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        # 4. Iterate through chunks
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    # Stream content to the Stage for this tool call
                    if tool_call_params.stage:
                        tool_call_params.stage.append_content(delta.content)
                    content += delta.content

                # Handle custom_content from delta
                delta_cc = getattr(delta, 'custom_content', None)
                if delta_cc:
                    # Set state from response CustomContent
                    state = delta_cc.get('state') if isinstance(delta_cc, dict) else getattr(delta_cc, 'state', None)
                    if state:
                        custom_content.state = state

                    # Propagate attachments to choice
                    attachments = delta_cc.get('attachments') if isinstance(delta_cc, dict) else getattr(delta_cc, 'attachments', None)
                    if attachments:
                        for att in attachments:
                            if isinstance(att, dict):
                                tool_call_params.choice.add_attachment(Attachment(**att))
                            else:
                                tool_call_params.choice.add_attachment(att)

                    # Optional: Stages propagation
                    if isinstance(delta_cc, dict):
                        cc_dict = delta_cc
                    elif hasattr(delta_cc, 'model_dump'):
                        cc_dict = delta_cc.model_dump()
                    elif hasattr(delta_cc, 'dict'):
                        cc_dict = delta_cc.dict()
                    else:
                        cc_dict = {}

                    stages = cc_dict.get('stages') if isinstance(cc_dict, dict) else None
                    if stages:
                        for stage_data in stages:
                            stage_index = stage_data.get('index', 0)
                            if stage_index in stages_map:
                                stage = stages_map[stage_index]
                            else:
                                stage = StageProcessor.open_stage(
                                    choice=tool_call_params.choice,
                                    name=stage_data.get('name', '')
                                )
                                stages_map[stage_index] = stage

                            # Propagate content
                            if stage_data.get('content'):
                                stage.append_content(stage_data['content'])

                            # Propagate attachments
                            if stage_data.get('attachments'):
                                for att_data in stage_data['attachments']:
                                    if isinstance(att_data, dict):
                                        stage.add_attachment(Attachment(**att_data))
                                    else:
                                        stage.add_attachment(att_data)

                            # Close stage if completed
                            if stage_data.get('status') == 'completed':
                                StageProcessor.close_stage_safely(stage)

        # 5. Ensure all stages are closed
        for stage in stages_map.values():
            StageProcessor.close_stage_safely(stage)

        # 6. Return Tool message with tool_call_id and custom_content
        return Message(
            role=Role.TOOL,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
            content=StrictStr(content),
            custom_content=custom_content,
        )

    def _prepare_messages(self, tool_call_params: ToolCallParams) -> list[dict[str, Any]]:
        # 1. Get prompt and propagate_history params from tool call
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments.get("prompt", "")
        propagate_history = arguments.get("propagate_history", False)

        # 2. Prepare empty messages array
        messages: list[dict[str, Any]] = []

        # 3. Collect the proper history if propagate_history is True
        if propagate_history:
            for i, message in enumerate(tool_call_params.messages):
                if (message.role == Role.ASSISTANT
                        and message.custom_content
                        and message.custom_content.state
                        and isinstance(message.custom_content.state, dict)
                        and self.name in message.custom_content.state):
                    # Add user message that is going before the assistant message
                    if i > 0 and tool_call_params.messages[i - 1].role == Role.USER:
                        prev_user_msg = tool_call_params.messages[i - 1]
                        messages.append({
                            "role": Role.USER.value,
                            "content": prev_user_msg.content or ""
                        })

                    # Add assistant message with refactored state
                    msg_copy = deepcopy(message)
                    msg_copy.custom_content.state = message.custom_content.state[self.name]
                    messages.append(msg_copy.dict(exclude_none=True))

        # 4. Add user message with prompt and custom_content
        user_msg: dict[str, Any] = {"role": Role.USER.value, "content": prompt}

        # Include custom_content from the last user message (for file attachments)
        for msg in reversed(tool_call_params.messages):
            if msg.role == Role.USER and msg.custom_content:
                user_msg["custom_content"] = msg.custom_content.dict(exclude_none=True)
                break

        messages.append(user_msg)

        return messages
