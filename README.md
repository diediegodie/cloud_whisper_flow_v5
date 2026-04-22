# Cloud Whisper Flow v5

Cloud Whisper Flow is a native desktop application (PySide6) for voice-to-text and optional translation.
It captures microphone audio, transcribes speech with Vosk (offline STT), and can translate output text with deep-translator.

This repository is organized for clean separation between UI, services, and infrastructure, with automated tests and Windows validation scripts.

## Key Features

- Native desktop UI with PySide6
- Main window with:
  - Start/Stop recording
  - Transcript output
  - Translation output
  - Source/target language selectors
  - Translation enable toggle
  - Auto-stop configuration
- Offline speech-to-text via Vosk model
- Optional text translation via deep-translator (GoogleTranslator)
- Config persistence via `config.json`
- Layered architecture with core/backend/frontend separation

## Tech Stack

- Python 3.10+
- PySide6
- Vosk
- sounddevice
- numpy
- deep-translator
- pytest + pytest-cov

## STT Model Requirement

The app expects a Vosk model directory configured in `config.json`.
Default path in this project:

```json
"vosk_model_path": "stt_model/vosk-model-small-pt-0.3/vosk-model-small-pt-0.3"
```

If you use a different model location, update `config.json` accordingly.

## Configuration

Edit `config.json`:

```json
{
  "source_language": "pt",
  "target_language": "en",
  "translation_enabled": true,
  "auto_stop_seconds": 8,
  "vosk_model_path": "stt_model/vosk-model-small-pt-0.3/vosk-model-small-pt-0.3"
}
```

Configuration keys:

- `source_language`: input language code (`pt`, `en`, `es`)
- `target_language`: translation target language code
- `translation_enabled`: enable/disable translation pipeline
- `auto_stop_seconds`: recording auto-stop delay (1 to 30)
- `vosk_model_path`: local Vosk model folder

## Run the Application

```bash
python main.py
```

## Usage Flow

1. Click **Start Recording**
2. Speak a short sentence
3. Click **Stop Recording**
4. Wait for transcription in the transcript box
5. If translation is enabled, translated text is shown

## Windows Validation

For release confidence, run native Windows checks using scripts in `scripts/`:

- `scripts/windows_run_validation.ps1`
- `scripts/windows_native_smoke.py`
- `scripts/windows_offline_translation_smoke.py`

## Architecture Notes

- `src/frontend`: Qt UI classes and UI event wiring
- `src/backend`: runtime services (audio capture, STT, translation)
- `src/core`: shared constants, config, and state definitions
- Frontend orchestrates service usage through a controller and worker thread strategy for responsiveness

## Known Considerations

- STT quality depends on microphone quality, speech clarity, and model size.
- PT-BR recognition is functional, but phrase-level accuracy can vary.
- Final acceptance for UI responsiveness should always be validated on native Windows.
