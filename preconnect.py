import asyncio
import logging

from livekit import rtc
from livekit.plugins import openai
from livekit.agents.utils.audio import AudioByteStream

logger = logging.getLogger("preconnect")
logger.setLevel(logging.DEBUG)

async def handle_pre_connect(model: openai.realtime.RealtimeModel, reader: rtc.ByteStreamReader):
    session = model.sessions[0]
    if session is None:
        return

    sample_rate = int(reader.info.attributes["sampleRate"])
    num_channels = int(reader.info.attributes["channels"])
    if sample_rate is None or num_channels is None:
        return

    audio_stream = AudioByteStream(sample_rate=sample_rate, num_channels=num_channels)
    if audio_stream is None:
        return

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
        try:
            await processor_task
        except asyncio.CancelledError:
            pass
