from __future__ import annotations

import os
from typing import List, Optional

import streamlit as st

CHAT_HISTORY_LIMIT = 8
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GEMINI_TEMPERATURE = 0.2
DEFAULT_GEMINI_MAX_OUTPUT_TOKENS = 2048
COACH_SYSTEM_PROMPT = (
    "You are a soccer performance analyst and coach. "
    "Use only the provided report JSON to answer. "
    "If a detail is missing, say you do not know. "
    "Be concise, practical, and specific. "
    "If you are cut off, end at a complete sentence."
)


def get_gemini_api_key() -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key
    secrets = getattr(st, "secrets", None)
    if not secrets:
        return None
    try:
        api_key = secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if api_key:
        return api_key
    try:
        return secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def gemini_prereq_error() -> Optional[str]:
    if not get_gemini_api_key():
        return "Set GEMINI_API_KEY in your environment or Streamlit secrets to enable chat."
    try:
        from google import genai  # noqa: F401
    except Exception:
        return "Install the google-genai package to enable chat."
    return None


def build_chat_prompt(report_json: str, history: List[dict], user_message: str) -> str:
    conversation_lines = []
    for msg in history:
        role = msg.get("role")
        prefix = "Assistant" if role == "assistant" else "User"
        content = msg.get("content", "")
        if content:
            conversation_lines.append(f"{prefix}: {content}")
    conversation_block = "\n".join(conversation_lines)
    parts = ["Report JSON:", report_json]
    if conversation_block:
        parts.extend(["Conversation:", conversation_block])
    parts.append(f"User: {user_message}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def generate_gemini_response(
    report_json: str,
    user_message: str,
    history: List[dict],
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> tuple[Optional[str], Optional[str]]:
    api_key = get_gemini_api_key()
    if not api_key:
        return None, "Missing GEMINI_API_KEY. Set it and restart the app."
    try:
        from google import genai
        from google.genai import types
    except Exception:
        return None, "Missing dependency: google-genai."
    history = history[-CHAT_HISTORY_LIMIT:]
    prompt = build_chat_prompt(report_json, history, user_message)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=COACH_SYSTEM_PROMPT,
    )
    try:
        client = st.session_state.get("gemini_client")
        if client is None:
            client = genai.Client(api_key=api_key)
            st.session_state["gemini_client"] = client
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        message = f"Gemini request failed: {exc.__class__.__name__}: {exc}"
        if "closed" in str(exc).lower():
            try:
                client = genai.Client(api_key=api_key)
                st.session_state["gemini_client"] = client
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc_retry:
                return None, f"Gemini request failed: {exc_retry.__class__.__name__}: {exc_retry}"
        else:
            return None, message
    text = getattr(response, "text", None)
    if text is None:
        return "", None

    def _finish_reason(resp) -> Optional[str]:
        candidates = getattr(resp, "candidates", None)
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        if hasattr(reason, "name"):
            return str(reason.name)
        return str(reason)

    finish_reason = _finish_reason(response)
    needs_continue = False
    if finish_reason:
        finish_upper = finish_reason.upper()
        if "MAX" in finish_upper or "LENGTH" in finish_upper:
            needs_continue = True

    if needs_continue:
        continuation_prompt = build_chat_prompt(
            report_json,
            history + [{"role": "assistant", "content": text}],
            "Continue from the last sentence without repeating.",
        )
        try:
            response2 = client.models.generate_content(
                model=model,
                contents=continuation_prompt,
                config=config,
            )
            continuation_text = getattr(response2, "text", None)
            if continuation_text:
                text = text.rstrip() + "\n\n" + continuation_text.strip()
        except Exception:
            pass

    return text.strip(), None
