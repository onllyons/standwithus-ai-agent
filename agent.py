import logging
import os
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent, llm, room_io
from livekit.plugins import noise_cancellation, elevenlabs, lemonslice

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)

# Temporary debug output for ElevenLabs env vars
print("ELEVEN_API_KEY:", bool(os.getenv("ELEVEN_API_KEY")))
print("ELEVEN_VOICE_ID:", os.getenv("ELEVEN_VOICE_ID"))

logger = logging.getLogger(__name__)
CHATBASE_API_URL = "https://www.chatbase.co/api/v1/chat"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required in your environment/.env")
    return value


def chatbase_messages(chat_ctx: llm.ChatContext) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in chat_ctx.items:
        if item.type != "message":
            continue
        if item.role not in ("user", "assistant"):
            continue
        text = item.text_content
        if not text:
            continue
        messages.append({"role": item.role, "content": text})
    return messages


class NoopLLM(llm.LLM):
    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options=agents.DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls=agents.NOT_GIVEN,
        tool_choice=agents.NOT_GIVEN,
        extra_kwargs=agents.NOT_GIVEN,
    ):
        del chat_ctx, tools, conn_options, parallel_tool_calls, tool_choice, extra_kwargs
        raise RuntimeError("NoopLLM.chat should not be called when Assistant.llm_node is overridden")


class Assistant(Agent):
    def __init__(self, *, conversation_id: str | None = None) -> None:
        super().__init__(
            instructions="""You are a helpful assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )
        self._chatbase_api_key = require_env("CHATBASE_API_KEY")
        self._chatbase_chatbot_id = require_env("CHATBASE_CHATBOT_ID")
        self._chatbase_conversation_id = conversation_id

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: object,
    ) -> str:
        del tools, model_settings
        messages = chatbase_messages(chat_ctx)
        if not messages:
            return "Hello. How can I help you today?"

        payload: dict[str, object] = {
            "chatbotId": self._chatbase_chatbot_id,
            "messages": messages,
            "stream": False,
        }
        if self._chatbase_conversation_id:
            payload["conversationId"] = self._chatbase_conversation_id

        headers = {
            "Authorization": f"Bearer {self._chatbase_api_key}",
            "Content-Type": "application/json",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=45)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(CHATBASE_API_URL, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        error_body = await response.text()
                        raise RuntimeError(
                            f"chatbase http {response.status}: {error_body[:500]}"
                        )
                    data = await response.json(content_type=None)
        except Exception:
            logger.exception("chatbase request failed")
            return "I am having trouble reaching the chatbot backend right now. Please try again."

        conversation_id = data.get("conversationId")
        if isinstance(conversation_id, str) and conversation_id.strip():
            self._chatbase_conversation_id = conversation_id

        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            logger.error("chatbase returned invalid response: %r", data)
            return "I could not generate a valid response right now. Please try again."

        return text.strip()


server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt="deepgram/nova-2",
        llm=NoopLLM(),
        tts=elevenlabs.TTS(
            api_key=os.getenv("ELEVEN_API_KEY"),
            voice_id="EXAVITQu4vr4xnSDxMaL",
            model="eleven_flash_v2_5",
        ),
        resume_false_interruption=False,
    )

    avatar = lemonslice.AvatarSession(
        agent_id=require_env("LEMONSLICE_AGENT_ID"),
    )

    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=Assistant(conversation_id=ctx.room.name),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(noise_cancellation=noise_cancellation.BVC()),
        ),
    )

    await session.generate_reply(instructions="Greet the user and offer your assistance.")


if __name__ == "__main__":
    agents.cli.run_app(server)
