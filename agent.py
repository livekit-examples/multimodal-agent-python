from __future__ import annotations

import os
import asyncio
import json
import logging
import uuid
from dotenv import load_dotenv
from dataclasses import asdict, dataclass

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobRequest,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai


load_dotenv(dotenv_path=".env.local")
sandbox = os.getenv("LIVEKIT_SANDBOX_ID")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


# The agent can be configured to only accept jobs from specific rooms
async def request(ctx: JobRequest):
    # In this case, when running in a sandbox we only want to join rooms
    # associated with that sandbox.
    if sandbox is not None:
        hash = sandbox.split('-')[-1]
        if ctx.room.name.startswith(f"sbx-{hash}"):
            return await ctx.accept()
        return await ctx.reject()
    return await ctx.accept()


@dataclass
class SessionConfig:
    openai_api_key: str
    instructions: str
    voice: str
    temperature: float = 0.8
    max_output_tokens: int | None = None
    modalities: list[str] | None = None

    def __post_init__(self):
        if self.modalities is None:
            self.modalities = self._modalities_from_string("text_and_audio")

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if k != "openai_api_key"}

    @staticmethod
    def _modalities_from_string(modalities: str) -> list[str]:
        modalities_map = {
            "text_and_audio": ["text", "audio"],
            "text_only": ["text"],
        }
        return modalities_map.get(modalities, ["text", "audio"])


def parse_session_config(data: dict) -> SessionConfig:
    return SessionConfig(
        openai_api_key=data.get("openai_api_key", ""),
        instructions=data.get("instructions", ""),
        voice=data.get("voice", ""),
        temperature=float(data.get("temperature", 0.8)),
        max_output_tokens=data.get("max_output_tokens") or 2048,
        modalities=SessionConfig._modalities_from_string(
            data.get("modalities", "text_and_audio")
        ),
    )


async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.Participant):
    metadata = json.loads(participant.metadata)
    config = parse_session_config(metadata)
    logger.info(f"starting multimodal agent with config: {config.to_dict()}")

    model = openai.realtime.RealtimeModel(
        api_key=config.openai_api_key,
        instructions=config.instructions,
        voice=config.voice,
        temparature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        modalities=config.modalities,
    )
    assistant = MultimodalAgent(model=model)
    assistant.start(ctx.room)

    session = model.sessions[0]
    session.default_conversation.item.create(
        llm.ChatMessage(
            role="user",
            content="Please begin the interaction with the user in a manner consistent with your instructions.",
        )
    )
    session.response.create()

    @ctx.room.on("participant_attributes_changed")
    def on_attributes_changed(
        changed_attributes: dict[str, str], participant: rtc.Participant
    ):
        new_config = parse_session_config(
            {**participant.attributes, **changed_attributes}
        )
        logger.info(f"participant attributes changed: {new_config.to_dict()}")
        session = model.sessions[0]
        session.session_update(
            instructions=new_config.instructions,
            # voice=assistant.inference_config().voice,
            temparature=new_config.temperature,
            max_output_tokens=new_config.max_output_tokens,
            # turn_detection=changed_attributes.get("turn_detection", participant.attributes.get("turn_detection", "server_vad")),
            modalities=new_config.modalities,
            # ).lower()
            # == "true",
            # vad_threshold=float(changed_attributes.get("vad_threshold", participant.attributes.get("vad_threshold", "0.5"))),
            # vad_silence_duration_ms=int(changed_attributes.get("vad_silence_duration_ms", participant.attributes.get("vad_silence_duration_ms", "200"))),
            # vad_prefix_padding_ms=int(changed_attributes.get("vad_prefix_padding_ms", participant.attributes.get("vad_prefix_padding_ms", "300"))),
        )
        if "instructions" in changed_attributes:
            session.default_conversation.item.create(
                llm.ChatMessage(
                    role="user",
                    content="Your instructions have changed. Please acknowledge this in a manner consistent with your new instructions. Do not explicitly mention the change in instructions.",
                )
            )
        session.response.create()

    async def send_transcription(
        ctx, participant, track_sid, segment_id, text, is_final=True
    ):
        transcription = rtc.Transcription(
            participant_identity=participant.identity,
            track_sid=track_sid,
            segments=[
                rtc.TranscriptionSegment(
                    id=segment_id,
                    text=text,
                    start_time=0,
                    end_time=0,
                    language="en",
                    final=is_final,
                )
            ],
        )
        await ctx.room.local_participant.publish_transcription(transcription)

    @session.on("response_done")
    def on_response_done(response: openai.realtime.RealtimeResponse):
        if response.status == "incomplete":
            message = "🚫 response incomplete"
        elif response.status == "failed":
            message = "⚠️ response failed"
        else:
            return

        local_participant = ctx.room.local_participant
        track_sid = next(
            (
                track.sid
                for track in local_participant.track_publications.values()
                if track.source == rtc.TrackSource.SOURCE_MICROPHONE
            ),
            None,
        )

        asyncio.create_task(
            send_transcription(
                ctx, local_participant, track_sid, str(uuid.uuid4()), message
            )
        )

    last_transcript_id = None

    # send three dots when the user starts talking. will be cleared later when a real transcription is sent.
    @session.on("input_speech_started")
    def on_input_speech_started():
        nonlocal last_transcript_id
        remote_participant = next(iter(ctx.room.remote_participants.values()), None)
        if not remote_participant:
            return

        track_sid = next(
            (
                track.sid
                for track in remote_participant.track_publications.values()
                if track.source == rtc.TrackSource.SOURCE_MICROPHONE
            ),
            None,
        )
        if last_transcript_id:
            asyncio.create_task(
                send_transcription(
                    ctx, remote_participant, track_sid, last_transcript_id, ""
                )
            )

        new_id = str(uuid.uuid4())
        last_transcript_id = new_id
        asyncio.create_task(
            send_transcription(
                ctx, remote_participant, track_sid, new_id, "…", is_final=False
            )
        )

    @session.on("input_speech_transcription_completed")
    def on_input_speech_transcription_completed(
        event: openai.realtime.InputTranscriptionCompleted,
    ):
        nonlocal last_transcript_id
        if last_transcript_id:
            remote_participant = next(iter(ctx.room.remote_participants.values()), None)
            if not remote_participant:
                return

            track_sid = next(
                (
                    track.sid
                    for track in remote_participant.track_publications.values()
                    if track.source == rtc.TrackSource.SOURCE_MICROPHONE
                ),
                None,
            )
            asyncio.create_task(
                send_transcription(
                    ctx, remote_participant, track_sid, last_transcript_id, ""
                )
            )
            last_transcript_id = None

    @session.on("input_speech_transcription_failed")
    def on_input_speech_transcription_failed(
        event: openai.realtime.InputTranscriptionFailed,
    ):
        nonlocal last_transcript_id
        if last_transcript_id:
            remote_participant = next(iter(ctx.room.remote_participants.values()), None)
            if not remote_participant:
                return

            track_sid = next(
                (
                    track.sid
                    for track in remote_participant.track_publications.values()
                    if track.source == rtc.TrackSource.SOURCE_MICROPHONE
                ),
                None,
            )

            error_message = "⚠️ Transcription failed"
            asyncio.create_task(
                send_transcription(
                    ctx,
                    remote_participant,
                    track_sid,
                    last_transcript_id,
                    error_message,
                )
            )
            last_transcript_id = None


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, worker_type=WorkerType.ROOM, request_fnc=request))

