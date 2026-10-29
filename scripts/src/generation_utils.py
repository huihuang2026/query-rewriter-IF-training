#!/usr/bin/env python3
"""Shared Gemini request retries and command-line argument validation."""
import argparse
import time

QUERY_FIELDS = ("query_with_user_instruction", "query_without_user_instruction")


class IncompleteGeneration(ValueError):
    def __init__(self, reason):
        super().__init__(f"Gemini finish reason: {reason}")
        self.retryable = reason in ("MAX_TOKENS", "OTHER", "NO_CANDIDATES", "FINISH_REASON_UNSPECIFIED", "None")
        self.reason = reason


def request_with_retries(client, attempts=3, **kwargs):
    for attempt in range(attempts):
        try:
            response = client.models.generate_content(**kwargs)
            candidates = response.candidates or []
            reason = candidates[0].finish_reason if candidates else "NO_CANDIDATES"
            reason = str(getattr(reason, "value", reason))
            if not candidates:
                feedback = getattr(response, "prompt_feedback", None)
                blocked = getattr(feedback, "block_reason", None)
                if blocked:
                    reason = str(getattr(blocked, "value", blocked))
            if reason != "STOP":
                raise IncompleteGeneration(reason)
            return response
        except Exception as error:
            transient = getattr(error, "code", None) in (408, 429, 500, 502, 503, 504)
            transient = transient or isinstance(error, TimeoutError) or type(error).__name__ in ("ReadTimeout", "ConnectTimeout")
            transient = transient or isinstance(error, IncompleteGeneration) and error.retryable
            if not transient or attempt == attempts - 1:
                raise
            delay = 2 ** (attempt + 1)
            if isinstance(error, IncompleteGeneration) and error.reason == "MAX_TOKENS":
                config = kwargs["config"]
                kwargs["config"] = config.model_copy(update={
                    "max_output_tokens": min((config.max_output_tokens or 8192) * 2, 32768)})
            print(f"Transient Gemini error ({error}); retry {attempt + 1}/{attempts - 1} in {delay}s", flush=True)
            time.sleep(delay)


def parse_limit(value):
    if value.strip().lower() in ("null", "none", ""):
        return None
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be NULL or a nonnegative integer")
    return number or None


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number
