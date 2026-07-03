CloudWhisper Lite – Development Standards

## 1. Architectural Principles

**Clear layer separation:**
- **Frontend (UI)**: PySide6, responsible only for rendering interface and capturing events.
- **Backend (Services)**: audio capture, STT, translation, business logic.
- **Core (Infrastructure)**: configuration, states, constants, utilities.

**Low coupling, high cohesion**: each module should have a single, well-defined responsibility.

**Well-defined interfaces**: frontend-backend communication via Qt signals/slots or service classes with public methods.

## 2. Clean Code Guidelines

- Follow PEP 8 (indentation, clear names, docstrings).
- Use type hints in all public functions.
- Name variables and functions descriptively (e.g., `start_recording()` instead of `doRec()`).
- Avoid code duplication (DRY principle).
- Keep functions short and focused on a single responsibility.

## 3. Modular Design

**Frontend/UI:**
- `main_window.py` → main window.

**Backend:**
- `audio.py` → audio capture.
- `stt_vosk.py` → transcription.
- `translator.py` → translation.

**Core:**
- `config.py` → configuration loading and persistence.
- `view_state.py` → state enum (IDLE, RECORDING, etc.).
- `constants.py` → fixed values (window size, delays, etc.).

## 4. Extensibility

**Component substitution:**
- **Translator**: encapsulate in `TranslatorService` → easily swap deep-translator for another API.
- **STT**: encapsulate in `SpeechToTextService` → replace Vosk with another model without affecting UI.
- **UI**: keep logic decoupled → possible migration to another framework without breaking backend.

**External configuration:**
- Languages, delays, and paths defined in `config.json`.
- Avoid hardcoded values in code.

## 5. Error Handling

- Use try/except for external calls (microphone, translation, STT).
- Convert errors to user-friendly UI messages.
- Never let unhandled exceptions reach the end user.

## 6. Testing Strategy

**Unit Tests:**
- Test each service in isolation (audio, STT, translation).
- Test state machine (valid and invalid transitions).

**Integration Tests:**
- Complete pipeline: audio → STT → translation.
- UI interacting with backend via signals.

**Acceptance Tests:**
- Real scenarios: recording and translation.
- Test on native Windows 10/11.

## 7. Responsiveness

- Heavy processing always in `QThread` or `QtConcurrent`.
- UI must never freeze during recording or translation.
- UI updates via signals/slots, never directly from threads.

**PySide6 Best Practices:**
- Reference official PySide6 documentation for signal/slot patterns and threading best practices.
- Follow PySide6 conventions to avoid common pitfalls and inconsistencies.
- Consult official docs for proper resource management and memory handling with fetch mcp


## 8. Versioning & Documentation

- Document each module with docstrings and internal README.
- Keep guidelines updated in `.docs/guidelines.md`.

## 9. Packaging & Distribution

- PyInstaller to generate executable.
- Test final binary on Windows before release.
- Include `config.json` and Vosk models in package.

## 10. Non-Negotiables

- Clear separation between frontend, backend, and core.
- No heavy processing on UI thread.
- External configuration in `config.json`.
- Mandatory tests before release.
- Clean, modular, and documented code.
