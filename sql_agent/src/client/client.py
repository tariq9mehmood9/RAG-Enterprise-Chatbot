import os
from typing import Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    UserInput,
)


class AgentClient:
    """Client for interacting with the agent service."""

    def __init__(
        self, base_url: str = "http://localhost:80", timeout: float | None = None
    ) -> None:
        """
        Initialize the client.

        Args:
            base_url (str): The base URL of the agent service.
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    async def ainvoke(self, message: str, thread_id: str | None = None) -> ChatMessage:
        """
        Invoke the agent asynchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            thread_id (str, optional): Thread ID for continuing a conversation

        Returns:
            AnyMessage: The response from the agent
        """
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                return ChatMessage.model_validate(response.json())
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def invoke(self, message: str, thread_id: str | None = None) -> ChatMessage:
        """
        Invoke the agent synchronously. Only the final message is returned.

        Args:
            message (str): The message to send to the agent
            thread_id (str, optional): Thread ID for continuing a conversation

        Returns:
            ChatMessage: The response from the agent
        """
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        response = httpx.post(
            f"{self.base_url}/invoke",
            json=request.model_dump(),
            headers=self._headers,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            return ChatMessage.model_validate(response.json())
        raise Exception(f"Error: {response.status_code} - {response.text}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """
        Create a feedback record for a run.

        This is a simple wrapper for the LangSmith create_feedback API, so the
        credentials can be stored and managed in the service rather than the client.
        See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/feedback",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise Exception(f"Error: {response.status_code} - {response.text}")
            response.json()

    def get_history(
        self,
        thread_id: str,
    ) -> ChatHistory:
        """
        Get chat history.

        Args:
            thread_id (str, optional): Thread ID for identifying a conversation
        """
        request = ChatHistoryInput(thread_id=thread_id)
        response = httpx.post(
            f"{self.base_url}/history",
            json=request.model_dump(),
            headers=self._headers,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            response_object = response.json()
            return ChatHistory.model_validate(response_object)
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
