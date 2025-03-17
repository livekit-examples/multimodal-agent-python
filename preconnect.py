import asyncio
import logging

from livekit import rtc
from livekit.plugins import openai
from livekit.agents import JobContext
from livekit.agents.utils.audio import AudioByteStream

logger = logging.getLogger("preconnect")
logger.setLevel(logging.DEBUG)

PRE_CONNECT_AUDIO_ATTRIBUTE = "lk.agent.pre-connect-audio"
PRE_CONNECT_AUDIO_BUFFER_STREAM = "lk.agent.pre-connect-audio-buffer"


def register_pre_connect_handler(ctx: JobContext, model: openai.realtime.RealtimeModel):
    logger.info("registering pre-connect audio handler")
    ctx.room.register_byte_stream_handler(PRE_CONNECT_AUDIO_BUFFER_STREAM, lambda reader, participant_identity: asyncio.create_task(handle_pre_connect(model, reader)))


async def handle_pre_connect(model: openai.realtime.RealtimeModel, reader: rtc.ByteStreamReader):
    if not model.sessions or model.sessions[0] is None:
        return

    session = model.sessions[0]
    
    try:
        sample_rate = int(reader.info.attributes["sampleRate"])
        num_channels = int(reader.info.attributes["channels"])
    except (KeyError, ValueError, TypeError):
        logger.error("Missing or invalid audio stream parameters")
        return

    audio_stream = AudioByteStream(sample_rate=sample_rate, num_channels=num_channels)
    input_buffer = session.input_audio_buffer
    
    try:
        async for chunk in reader:
            logger.debug(f"Received chunk of {len(chunk)} bytes")
            for frame in audio_stream.write(chunk):
                input_buffer.append(frame)
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
    finally:
        logger.debug("Flushing audio buffer")
        for frame in audio_stream.flush():
            input_buffer.append(frame)
