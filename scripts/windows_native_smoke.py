"""Native Windows smoke validation for backend runtime behavior.

Run this from project root on native Windows after setting up dependencies:
    python scripts/windows_native_smoke.py --attempts 10 --duration-seconds 3
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

try:
    import winsound
except ImportError:
    winsound = None

import numpy as np
import sounddevice as sd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.audio import AudioCaptureService, AudioCaptureError
from src.backend.stt_vosk import SpeechToTextService, SpeechToTextError
from src.backend.translator import TranslatorService
from src.core.config import ConfigManager


def _project_root() -> Path:
    return PROJECT_ROOT


def _load_runtime_config() -> dict:
    config_path = _project_root() / "config.json"
    manager = ConfigManager(config_path=config_path)
    return manager.get_all()


def _resolve_model_path(value: str) -> Path:
    model_path = Path(value)
    if not model_path.is_absolute():
        model_path = _project_root() / model_path
    return model_path


def _check_audio_devices() -> tuple[bool, str]:
    devices = sd.query_devices()
    input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
    default_input = sd.default.device[0]

    if not input_devices:
        return False, "No input audio devices detected by sounddevice."
    if default_input == -1:
        return False, "Default input audio device is not configured."

    return True, (
        f"Input devices: {len(input_devices)} | "
        f"Default input index: {default_input}"
    )


def _run_translation_smoke(translator: TranslatorService) -> tuple[bool, str]:
    source = "ola mundo"
    translated = translator.translate(source)
    if not isinstance(translated, str) or translated.strip() == "":
        return False, "Translator returned empty/non-string output."
    return True, f"translate('{source}') -> '{translated}'"


def _run_recording_cycle(
    attempt: int,
    duration_seconds: int,
    audio_service: AudioCaptureService,
    stt_service: SpeechToTextService,
    translator: TranslatorService,
) -> tuple[bool, str]:
    beep = getattr(winsound, "Beep", None)
    if callable(beep):
        beep(1200, 300)

    print(
        f"START: Attempt {attempt} is beginning. Speak now for "
        f"{duration_seconds} seconds...",
        flush=True,
    )
    try:
        audio_service.start_recording()
        time.sleep(duration_seconds)
        audio = audio_service.stop_recording()
    except AudioCaptureError as exc:
        return False, f"Attempt {attempt}: audio failure -> {exc}"

    if audio.size == 0:
        return False, f"Attempt {attempt}: captured empty audio buffer."

    try:
        text = stt_service.transcribe(audio)
    except SpeechToTextError as exc:
        return False, f"Attempt {attempt}: STT failure -> {exc}"

    if text.strip() == "":
        return False, f"Attempt {attempt}: STT returned empty text."

    translated = translator.translate(text)
    if translated.strip() == "":
        return False, f"Attempt {attempt}: translation returned empty text."

    summary = {
        "attempt": attempt,
        "samples": int(audio.size),
        "stt_text": text,
        "translated": translated,
    }
    return True, json.dumps(summary, ensure_ascii=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Native Windows smoke validation")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--duration-seconds", type=int, default=3)
    parser.add_argument(
        "--skip-recording",
        action="store_true",
        help="Skip microphone recording loop and run only device/STT/translator checks.",
    )
    parser.add_argument(
        "--skip-audio-device-check",
        action="store_true",
        help="Skip audio device availability check (useful for non-Windows dry-runs).",
    )
    args = parser.parse_args()

    print("=== Windows Native Smoke Validation ===")
    print(f"Platform: {platform.platform()}")
    if platform.system().lower() != "windows":
        print("WARNING: This script is intended for native Windows.")

    config = _load_runtime_config()
    model_path = _resolve_model_path(config["vosk_model_path"])

    print(f"Using model path: {model_path}")

    try:
        stt_service = SpeechToTextService(str(model_path))
    except Exception as exc:
        print(f"FAIL: Could not initialize STT service -> {exc}")
        return 2

    translator = TranslatorService(
        config["source_language"],
        config["target_language"],
    )
    audio_service = AudioCaptureService()

    if args.skip_audio_device_check:
        print("SKIP: Audio device availability check skipped by flag.")
    else:
        ok, msg = _check_audio_devices()
        print(("PASS" if ok else "FAIL") + f": Audio devices -> {msg}")
        if not ok:
            return 3

    ok, msg = _run_translation_smoke(translator)
    print(("PASS" if ok else "FAIL") + f": Translation smoke -> {msg}")
    if not ok:
        return 4

    empty_text = stt_service.transcribe(np.array([], dtype=np.float32))
    if empty_text != "":
        print("FAIL: STT empty-input contract is broken.")
        return 5
    print("PASS: STT empty-input contract validated.")

    if args.skip_recording:
        print("SKIP: Recording loop skipped by flag.")
        return 0

    print(
        f"Running {args.attempts} recording attempts, "
        f"each with {args.duration_seconds}s capture..."
    )
    failures = 0

    for attempt in range(1, args.attempts + 1):
        ok, msg = _run_recording_cycle(
            attempt,
            args.duration_seconds,
            audio_service,
            stt_service,
            translator,
        )
        print(("PASS" if ok else "FAIL") + f": {msg}")
        if not ok:
            failures += 1

    if failures > 0:
        print(f"RESULT: FAILED ({failures}/{args.attempts} attempts failed)")
        return 6

    print("RESULT: PASS (all recording attempts succeeded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
