"""Application state machine for Cloud Whisper Flow."""

from enum import Enum, auto


class ViewState(Enum):
    """Enumeration of application states in the state machine.

    States represent different operational modes of the application:
    - IDLE: Application ready, no recording or processing.
    - RECORDING: Audio is being captured from the microphone.
    - PROCESSING: Audio is being transcribed or translated.
    - ERROR: An error occurred during recording or processing.

    Valid transitions are defined in the state machine's update logic.
    """

    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()
    ERROR = auto()
