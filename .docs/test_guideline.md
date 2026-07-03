CloudWhisper Lite – Test Guideline

## 1. Goal
Define a complete test strategy to validate Cloud Whisper Flow before and after frontend implementation, covering all repository modules, critical flows, error handling, and platform-specific behavior (WSL and native Windows).

## 2. Test Stages

1. Stage A: Core and Backend Verification (before frontend)
2. Stage B: Frontend Integration Verification (after frontend)
3. Stage C: Full End-to-End Validation (native Windows required)
4. Stage D: Regression and Release Gate

## 3. Repository Coverage Matrix

This checklist maps tests to every relevant file in the repository.

### 3.1 Core

- [x] src/core/constants.py
  - [x] Constant names and values are correct
  - [x] Window dimensions and titles match product requirements
  - [x] Default language and timing values match config defaults
- [x] src/core/view_state.py
  - [x] Enum contains IDLE, RECORDING, PROCESSING, ERROR
  - [x] Enum values are unique
  - [x] State comparisons work in logic and tests
- [x] src/core/config.py
  - [x] Loads valid config JSON
  - [x] Handles missing file gracefully
  - [x] Handles invalid JSON gracefully
  - [x] Save persists changes correctly
  - [x] get, set, update, get_all work correctly
  - [x] Singleton get_config returns stable instance
- [x] src/core/__init__.py
  - [x] Module import smoke test

### 3.2 Backend

- [x] src/backend/audio.py
  - [x] start_recording opens stream and sets recording state
  - [x] stop_recording returns mono float32 ndarray
  - [x] stop without start raises AudioCaptureError
  - [x] start while recording raises AudioCaptureError
  - [x] Empty capture returns empty array
  - [x] Multiple sessions do not leak stale buffer
  - [x] Callback buffer writes are thread-safe
- [x] src/backend/stt_vosk.py
  - [x] Constructor validates model path (FileNotFoundError when missing)
  - [x] is_ready reports model status
  - [x] transcribe(empty array) returns empty string
  - [x] float32 audio conversion to int16 PCM is valid
  - [x] Recognizer chunk feeding works and returns final text
  - [x] Transcription exceptions are wrapped/logged properly
- [x] src/backend/translator.py
  - [x] Constructor rejects empty language codes (ValueError)
  - [x] translate(empty or whitespace) returns original text unchanged
  - [x] Successful translation returns translated text
  - [x] Network/API error returns original text (graceful degradation)
  - [x] set_languages updates source/target correctly
  - [x] get_source_language and get_target_language return current values
- [x] src/backend/__init__.py
  - [x] Module import smoke test

### 3.3 Frontend

- [ ] src/frontend/main_window.py
  - [ ] Main window has two text boxes (Transcription, Translation)
  - [ ] Buttons REC/STOP exist and are wired
  - [ ] Translation controls (toggle/source/target) are wired
  - [ ] UI state follows ViewState transitions
- [ ] src/frontend/__init__.py
  - [ ] Module import smoke test

### 3.4 Entry and Configuration

- [ ] main.py
  - [ ] Application starts without crash
  - [ ] Services are initialized and connected
  - [ ] Shutdown path releases resources cleanly
- [ ] config.json
  - [x] Required keys exist
  - [x] Value types are correct
  - [x] vosk_model_path points to a valid model directory
  - [x] Values are consistent with constants defaults
- [ ] requirements.txt
  - [x] All required runtime dependencies are present

### 3.5 Documentation and Packaging Signals

- [ ] .docs/guideline.md
  - [x] Acceptance tests map to UX requirements
- [ ] .docs/standards.md
  - [x] Test strategy aligns with architecture and thread constraints
- [ ] .docs/development_log.md
  - [x] Test status updates are reflected at each implementation milestone

## 4. Stage A – Before Frontend (Automation First)

Run these first in WSL with pytest and mocks.

### 4.1 Core Unit Tests

- [x] constants value tests
- [x] ViewState enum existence and semantics
- [x] ConfigManager load/save/get/set/update/get_all tests
- [x] ConfigManager error handling (missing/invalid JSON)

### 4.2 Backend Unit Tests (Mock External Dependencies)

- [x] audio.py tests with mocked sounddevice.InputStream
- [x] stt_vosk.py tests with mocked Model and KaldiRecognizer
- [x] translator.py tests with mocked GoogleTranslator

### 4.3 Backend Integration Tests (No UI)

- [x] Audio service output format contract: mono float32 array
- [x] STT service accepts audio contract from audio service
- [x] Translator service accepts STT output text
- [x] Pipeline test: audio array -> STT text -> translated text (with mocks where needed)

### 4.4 Negative and Recovery Tests

- [x] Audio device unavailable
- [x] Invalid Vosk model path
- [x] Translation API/network failure
- [x] Verify no hard crash and clear logging in each case

## 5. Stage B – After Frontend Is Implemented

### 5.1 Main Window Functional Tests

- [ ] REC starts recording and toggles button state
- [ ] STOP ends recording and triggers STT processing
- [ ] Translation updates according to toggle and selected language pair
- [ ] UI remains responsive during processing

### 5.3 State Machine and UX Behavior

- [ ] Allowed transitions only: IDLE -> RECORDING -> PROCESSING -> IDLE
- [ ] Error path transitions to ERROR and then recovers to IDLE
- [ ] Buttons are disabled/enabled according to active state

### 5.4 Threading and Responsiveness

- [ ] Heavy operations run off UI thread (QThread/QtConcurrent)
- [ ] No UI freeze during recording, STT, or translation
- [ ] UI updates happen via signals/slots only

## 6. Stage C – End-to-End Manual Validation

### 6.1 WSL Manual Smoke Tests

- [x] Import and instantiate all core/backend services
- [x] Run non-UI pipeline with sample data
- [x] Validate basic behavior and logs
- [ ] Note: treat WSL audio behavior as preliminary only

### 6.2 Native Windows Mandatory Tests

- [x] 10 consecutive recording attempts without failure
- [ ] STT produces expected text quality for PT-BR speech
- [x] Translation works online
- [x] Translation failure path works offline (graceful fallback)
- [ ] Microphone and translation errors do not freeze app

#### Native Windows Test Runbook (Developer Procedure)

Use this exact procedure on native Windows (PowerShell) to run reproducible validation before release.

1. Open PowerShell
2. Go to project folder on Windows (example: `C:\dev\cloud_whisper_flow_v5`)
3. Create and activate virtual environment
4. Install dependencies from requirements
5. Run automated test suite
6. Run backend smoke checks (real model and translation)
7. Run manual microphone and UX acceptance checks

##### 6.2.1 Environment Setup (PowerShell)

```powershell
Set-Location C:\dev\cloud_whisper_flow_v5
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the `py` launcher is not available on the machine, use the installed interpreter path directly, for example:

```powershell
& C:\Users\diiie\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
``` 

##### 6.2.2 Automated Validation on Windows

```powershell
pytest -q tests\core tests\backend tests\project
pytest --cov=src --cov-branch --cov-report=term-missing -q tests\core tests\backend tests\project
```

Pass criteria:
- All tests pass
- Coverage report has no unexpected drops in core/backend modules

##### 6.2.3 Backend Smoke Check (Real Runtime)

```powershell
python - <<'PY'
from src.backend.translator import TranslatorService
from src.backend.stt_vosk import SpeechToTextService
import numpy as np

tr = TranslatorService('pt', 'en')
print('translate("ola mundo") =>', tr.translate('ola mundo'))

stt = SpeechToTextService('stt_model/vosk-model-small-pt-0.3/vosk-model-small-pt-0.3')
print('stt ready:', stt.is_ready())
print('transcribe(empty) =>', repr(stt.transcribe(np.array([], dtype=np.float32))))
PY
```

Pass criteria:
- Translation returns a non-empty English result for `ola mundo`
- STT model loads with `is_ready() == True`

##### 6.2.4 Microphone Device Availability Check

```powershell
python - <<'PY'
import sounddevice as sd
print('device count:', len(sd.query_devices()))
print('default devices:', sd.default.device)
PY
```

Pass criteria:
- At least one input device is available
- Default input device is not `-1`

##### 6.2.5 Manual Acceptance: 10 Consecutive Real Recordings

Execute 10 full cycles without restarting app/process:
1. Start recording
2. Speak a short PT-BR sentence (2 to 5 seconds)
3. Stop recording
4. Confirm STT output appears and is reasonable
5. Trigger translation and confirm output appears

Track each attempt:

- [x] Attempt 1
- [x] Attempt 2
- [x] Attempt 3
- [x] Attempt 4
- [x] Attempt 5
- [x] Attempt 6
- [x] Attempt 7
- [x] Attempt 8
- [x] Attempt 9
- [x] Attempt 10

Pass criteria:
- No crashes or freezes
- Recording starts/stops correctly in all 10 attempts
- STT returns text in all attempts
- Translation returns output in all attempts when online

##### 6.2.6 Offline Translation Fallback Check

1. Disable internet (airplane mode or disconnect network)
2. Trigger translation for existing PT-BR text
3. Verify application does not crash and remains responsive
4. Verify graceful fallback behavior (original text preserved)

Pass criteria:
- No freeze or crash
- Error path is handled gracefully

##### 6.2.7 Result Logging Requirement

After running native Windows tests:
1. Update `.docs/development_log.md` testing checklist
2. Record any failing scenario and exact reproduction steps
3. Do not mark Stage C complete until all mandatory checks pass

##### 6.2.8 Native Windows Evidence (2026-03-25)

- Execution context: launched from WSL via `powershell.exe`, using a Windows copy at `C:\dev\cloud_whisper_flow_v5`
- Source revision: `469ff31`
- Machine: `DIEDIEGODIE`
- Platform: `Windows-11-10.0.26200-SP0`
- Native interpreter: `C:\Users\diiie\AppData\Local\Programs\Python\Python312\python.exe`
- Native project environment: `C:\dev\cloud_whisper_flow_v5\.venv`
- Automated suite result: `50 passed in 1.07s`
- Coverage result: `50 passed in 3.03s`, total coverage `96%`
- Audio device check: `9` input devices detected, default input index `1`
- Online translation smoke: `translate("ola mundo") -> "Hello World"`
- Offline translation fallback: forced network failure preserved original text `"ola mundo"` and returned PASS
- Real microphone evidence: cue-guided run logged `10/10` successful captures with non-empty STT and non-empty translation outputs in `C:\dev\cloud_whisper_flow_v5\native_mic_validation.log`
- STT quality note: recognition was usable in all ten attempts, but some outputs were imperfect (`"olá do teste"`, `"olá bom domingo"`, `"o lado do teste"`), so PT-BR quality remains open for stricter review
- Frontend-dependent items remain open: single-window UI freeze/responsiveness and stricter PT-BR quality checks are pending

## 7. Stage D – Regression Suite

Run on each feature merge and before release.

- [ ] Full unit test suite
- [ ] Service integration suite
- [ ] Frontend functional suite (when available)
- [ ] Critical manual smoke tests (Windows)

## 8. Suggested Test Tooling

- [ ] pytest
- [ ] unittest.mock or pytest-mock
- [ ] pytest-cov for coverage reports
- [ ] Optional: pytest-qt after frontend is implemented

## 9. Mocking Guidelines

- [ ] Mock sounddevice for deterministic CI tests
- [ ] Mock Vosk recognizer/model for fast STT unit tests
- [ ] Mock deep-translator network calls for reliable translation tests
- [ ] Keep a small number of real integration tests separate from unit tests

## 10. Release Gate Criteria

The app is considered ready only when all items below are green:

- [ ] Stage A complete
- [ ] Stage B complete
- [ ] Stage C complete on native Windows
- [ ] No critical or high severity defects open
- [ ] Development log updated with latest test results

## 11. Notes for Current Project State

- Core and backend services are implemented and should be validated immediately using Stage A.
- Frontend files are still pending implementation and should be validated with Stage B only after the frontend is fully implemented.
- Because development is in WSL, use automation as primary signal and native Windows as final acceptance signal for audio and UI behavior.
