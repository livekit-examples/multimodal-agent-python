from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from livekit.agents.utils.audio import AudioByteStream

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.DEBUG)


async def entrypoint(ctx: JobContext):    
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        instructions=(
            "You are a voice assistant created by LiveKit. Your interface with users will be voice. "
            "You should use short and concise responses, and avoiding usage of unpronouncable punctuation. "
            "You were created as a demo to showcase the capabilities of LiveKit's agents framework."
        ),
        modalities=["audio", "text"],
    )

    # create a chat context with chat history, these will be synchronized with the server
    # upon session establishment
    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        text="Context about the user: you are talking to a software engineer who's building voice AI applications."
        "Greet the user with a friendly greeting and ask how you can help them today.",
        role="assistant",
    )

    agent = MultimodalAgent(
        model=model,
        chat_ctx=chat_ctx,
    )

    if participant.attributes.get("lk.agent.pre-connect-audio"):
        ctx.room.register_byte_stream_handler("lk.pre-connect-audio-buffer", lambda reader, participant_identity: asyncio.create_task(handle_pre_connect(agent, model.sessions[0], reader)))
    else:
        ctx.room.unregister_byte_stream_handler("lk.pre-connect-audio-buffer")

    agent.start(ctx.room, participant)

    # to enable the agent to speak first
    # agent.generate_reply()


async def handle_pre_connect(agent: MultimodalAgent, session: openai.realtime.RealtimeModel.Session, reader: rtc.ByteStreamReader):
    audio_stream = AudioByteStream(sample_rate=24000, num_channels=1, samples_per_channel=480)
    audio_queue = asyncio.Queue()
    input_buffer = session.input_audio_buffer
    
    async def process_queue():
        while True:
            chunk = await audio_queue.get()
            try:
                for frame in audio_stream.write(chunk):
                    input_buffer.append(frame)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
            finally:
                audio_queue.task_done()

    processor_task = asyncio.create_task(process_queue())
    
    try:
        async for chunk in reader:
            logger.debug(f"Received chunk of {len(chunk)} bytes")
            await audio_queue.put(chunk)
    finally:
        processor_task.cancel()
        for frame in audio_stream.flush():
            input_buffer.append(frame)
        session.commit_audio_buffer()
        agent.generate_reply()
        try:
            await processor_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )
