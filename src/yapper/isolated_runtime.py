"""SDK-independent JSON server, importable by older isolated Python runtimes."""

import contextlib
import json
import logging
import sys

MAX_MESSAGE = 65536


def serve(session):
    logging.disable(logging.INFO)
    while line := sys.stdin.readline(MAX_MESSAGE + 1):
        try:
            if len(line) > MAX_MESSAGE or not line.endswith("\n"):
                raise ValueError("Runtime request is too large or incomplete.")
            request = json.loads(line)
            if request["action"] not in ("load", "synthesize"):
                raise ValueError("Unknown isolated model action.")
            with contextlib.redirect_stdout(sys.stderr):
                result = getattr(session, request["action"])(request)
            response = {"ok": True, "result": result}
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, ensure_ascii=True), flush=True)
