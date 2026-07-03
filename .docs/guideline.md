CloudWhisper Lite – Guideline Document

## 1. Product Goal

Build a lightweight desktop app that:

- Captures user voice (PT-BR).
- Transcribes to text.
- Translates output when translation is enabled.
- Displays original and translated text in a simple interface.

## 2. Required UX

### Main Window
- **TextBox 1 (Transcription)**: displays text captured via STT.
- **TextBox 2 (Translation)**: displays text translation.
- **Buttons**:
    - REC/STOP → start/stop recording.
- **Controls**:
    - Translation toggle.
    - Source and target language selectors.
    - Auto-stop seconds.

## 3. Scope v1.0

**Included**:
- Offline STT with Vosk.
- Audio capture with sounddevice.
- Optional translation with deep-translator (free Google Translate).
- Native UI with PySide6.
- Basic configuration in config.json.

**Excluded** (for future versions):
- Automatic text injection at cursor.
- Inter-process communication.
- Continuous real-time translation.

## 4. Runtime and Platform
- Python 3.10+
- PySide6 for UI.
- Primary target: Windows 10/11.
- Development allowed on WSL Ubuntu, but final testing must be on native Windows.

## 5. Technical Stack Dependencies
- PySide6>=6.10.0 – native UI.
- vosk>=0.3.45 – offline STT.
- sounddevice>=0.4.6 – audio capture.
- numpy>=1.24.0 – audio processing.
- deep-translator==1.11.4 – on-demand translation.

**Installation**:
```bash
pip install PySide6 vosk sounddevice numpy deep-translator
```

## 6. Functional Flow

### Main Window Flow
1. User opens app.
2. Clicks REC → audio captured.
3. Transcribed text appears in TextBox 1.
4. If translation is enabled, translation appears in TextBox 2.
5. User can copy translated text to another app.

## 7. State Machine
States: `IDLE`, `RECORDING`, `PROCESSING`, `ERROR`

**Transitions**:
- IDLE → RECORDING
- RECORDING → PROCESSING
- PROCESSING → IDLE
- PROCESSING → ERROR
- ERROR → IDLE

## 8. Configuration

**config.json**:
```json
{
    "source_language": "pt",
    "target_language": "en",
    "translation_enabled": true,
    "auto_stop_seconds": 8,
    "vosk_model_path": "model/vosk-model-small-pt-0.3"
}
```

## 9. Project Structure
```
cloud_whisper_lite/
    main.py
    src/
        frontend/
            main_window.py
        backend/
            audio.py
            stt_vosk.py
            translator.py
        core/
            config.py
            view_state.py
    config.json
    requirements.txt
    model/
        vosk-model-small-pt-0.3/
```

## 10. Coding Guidelines
- Follow PEP 8.
- Use docstrings in public functions.
- Heavy processing in threads (QThread).
- Minimalist and responsive UI.
- Errors handled with clear messages (no crashes).

## 11. Testing Checklist
- [ ] Recording and transcription work in 10 consecutive attempts.
- [ ] Translation on demand works with and without internet.
- [ ] Microphone or translation errors do not freeze the app.

## 12. Packaging

**PyInstaller**:
```bash
pyinstaller --onefile --windowed --name CloudWhisperLite main.py
```

## 13. Non-Negotiables
- Two mandatory text boxes (Transcription + Translation).
- UI must be native PySide6.
- Backend must be separated from UI.
- App must be functional on Windows 10/11.
