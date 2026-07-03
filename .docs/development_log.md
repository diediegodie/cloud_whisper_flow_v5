CloudWhisper Lite - Development Log

## Current Version
- v6 - Single-window frontend architecture
- Status date: 2026-04-22
- Current test status: 50 tests passing in WSL (`venv/bin/python -m pytest -q tests`)

---

## Current Product State
- Main app runs as a single window only; compact mode has been removed.
- UI uses a controller + worker model (`FrontendController` + `ProcessingWorker`) with heavy processing on `QThread`.
- Transcript box is editable by the user.
- Translation box is read-only.
- STT output is appended to transcript content (does not replace previous lines).
- A `Clear` button clears transcript content.
- A `Translate` button translates current transcript content without requiring microphone input.
- Runtime source/target language changes apply immediately to the live translator service.
- WSL startup fallback forces `QT_QPA_PLATFORM=xcb` when running under WSL with no explicit Qt platform set.

---

## Implemented Features
- [ADDED] Project structure with `src/frontend`, `src/backend`, and `src/core`.
- [ADDED] Entry point in `main.py`.
- [ADDED] Documentation folder `.docs/`.
- [ADDED] `src/core/constants.py` with shared app/UI/config defaults.
- [ADDED] `src/core/view_state.py` with `ViewState` (`IDLE`, `RECORDING`, `PROCESSING`, `ERROR`).
- [ADDED] `src/core/config.py` with `ConfigManager`, load/save support, and singleton `get_config()`.
- [ADDED] `config.json` with default app settings.
- [ADDED] `src/backend/audio.py` (`AudioCaptureService`) for callback-based microphone capture.
- [ADDED] `src/backend/stt_vosk.py` (`SpeechToTextService`) for Vosk STT integration.
- [ADDED] `src/backend/translator.py` (`TranslatorService`) for translation with runtime language updates and fallback behavior.
- [ADDED] Main frontend in `src/frontend/main_window.py` (`MainWindow`, `FrontendController`, `ProcessingWorker`).
- [ADDED] `UI_BUTTON_TRANSLATE` and `UI_BUTTON_CLEAR_TRANSCRIPT` constants.
- [ADDED] Full pytest structure for core/backend/project checks and integration coverage.
- [ADDED] Windows-native smoke/validation scripts under `scripts/`.
- [CHANGED] `config.json` Vosk model path corrected to real local model directory.
- [CHANGED] WSL bootstrap behavior in `main.py` to avoid Qt/WSLg Wayland startup hangs.
- [REMOVED] `src/frontend/compact_window.py` and compact-window wiring.
- [REMOVED] `tests/project/test_main_bootstrap.py` (temporary bootstrap experiment test).

---

## Recent Fixes
- [FIXED] Startup config-loading side effects: `_load_config_values()` now uses `QSignalBlocker` to avoid accidental config writes during UI initialization.
- [FIXED] Transcript duplication on manual Translate: manual translation no longer appends already-present transcript text to itself on repeated clicks.
- [FIXED] Runtime language switching: changing source/target language in UI now updates active `TranslatorService` immediately via `set_languages()`.

---

## In Progress
- Frontend-focused tests in `tests/frontend/` (controller state transitions and window behavior).

---

## Backlog
- Auto-stop timer implementation.
- Frontend responsiveness validation in native Windows execution.
- Additional UI/UX consistency checks for long sessions and repeated interactions.

---

## Testing Checklist
- [x] Core and backend unit tests pass in WSL.
- [x] Backend failure-path tests pass in WSL.
- [x] Backend integration tests pass in WSL.
- [x] WSL non-UI smoke validation (translator live call, STT model load) passes.
- [x] Audio recording works for 10 consecutive attempts (native Windows validation).
- [x] STT output appends correctly to transcript without replacing previous content.
- [x] Manual Translate button works without microphone input.
- [x] Transcript does not duplicate on repeated Translate clicks.
- [x] Runtime language change takes effect immediately without app restart.
- [x] Translation works with and without internet (fallback verified).
- [ ] STT always returns text without freezing UI.
- [ ] Auto-stop timer stops recording automatically.
- [ ] Error UX is fully validated for all expected failure cases.

---

## Native Windows Validation Evidence (2026-03-25)
- [CHANGED] Validation executed from WSL into isolated copy `C:\dev\cloud_whisper_flow_v5`.
- [CHANGED] Revision under test: `469ff31`.
- [CHANGED] Platform context: `Windows-11-10.0.26200-SP0`.
- [CHANGED] Native Python used: `C:\Users\diiie\AppData\Local\Programs\Python\Python312\python.exe`.
- [CHANGED] Dependencies installed only in `C:\dev\cloud_whisper_flow_v5\.venv`.
- [CHANGED] Automated suite result: `50 passed in 1.07s`.
- [CHANGED] Coverage result: `96%` across `src`.
- [CHANGED] Device check: `9` input devices, default input index `1`.
- [CHANGED] Online translation smoke passed: `translate("ola mundo") -> "Hello World"`.
- [CHANGED] Offline translation fallback passed with forced proxy failure and original text preserved.
- [CHANGED] Cue-guided mic validation passed `10/10` attempts with non-empty STT and translation output.
- [CHANGED] STT quality is usable but imperfect for short PT-BR phrases; strict quality acceptance remains open.

---

## Development Standards Reminder
- Follow PEP 8 and type hints.
- Keep frontend, backend, and core separated.
- Use `QThread` for heavy tasks (audio/STT/translation).
- Update this log after every functionality change.
- Never hardcode operational values; use `config.json` or constants.

---

## Notes
- Keep using tags: `[ADDED]`, `[CHANGED]`, `[REMOVED]`, `[FIXED]`.
- In this WSL environment, `sounddevice` may report no input devices (`default input: [-1, -1]`), so audio acceptance should continue to be validated from native Windows when required.
