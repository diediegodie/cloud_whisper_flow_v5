"""Force a translator network failure and verify graceful fallback behavior."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.translator import TranslatorService


def main() -> int:
    original_text = "ola mundo"

    os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9"

    translator = TranslatorService("pt", "en")
    result = translator.translate(original_text)

    print(f"OFFLINE_TRANSLATION_INPUT={original_text!r}")
    print(f"OFFLINE_TRANSLATION_RESULT={result!r}")

    if result != original_text:
        print("FAIL: Offline fallback did not preserve the original text.")
        return 1

    print("PASS: Offline translation fallback preserved the original text.")
    return 0


if __name__ == "__main__":
    sys.exit(main())