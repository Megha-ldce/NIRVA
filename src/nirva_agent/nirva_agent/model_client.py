#!/usr/bin/env python3
"""model_client.py - OpenAI-compatible wrapper for Nebius Token Factory (text tool-calling + vision).

Every call is appended to a JSONL log (default ~/nirva_llm_calls.jsonl, override with
NIRVA_LLM_LOG) with model, latency, token usage and the tool chosen. That file is the
evidence for the hackathon write-up and the demo's latency/token numbers.
"""
import json
import os
import time

from openai import OpenAI

LOG_PATH = os.environ.get('NIRVA_LLM_LOG', os.path.expanduser('~/nirva_llm_calls.jsonl'))


class ModelClient:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=2)
        self.base_url = base_url

    # ------------------------------------------------------------------ logging
    def _log(self, kind: str, model: str, t0: float, response=None, tool: str = None, error: str = None):
        usage = getattr(response, 'usage', None)
        rec = {
            'ts': round(time.time(), 3),
            'kind': kind,
            'model': model,
            'endpoint': self.base_url,
            'latency_s': round(time.time() - t0, 3),
            'prompt_tokens': getattr(usage, 'prompt_tokens', None),
            'completion_tokens': getattr(usage, 'completion_tokens', None),
            'total_tokens': getattr(usage, 'total_tokens', None),
            'tool': tool,
            'error': error,
        }
        try:
            with open(LOG_PATH, 'a') as f:
                f.write(json.dumps(rec) + '\n')
        except OSError:
            pass  # logging must never break the robot

    # --------------------------------------------------------------------- text
    def plan(self, model: str, system_prompt: str, user_prompt: str, tools: list):
        """Tool-calling for skill selection. Returns (tool_name, args_dict, raw_response)."""
        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=tools,
                tool_choice="auto",
                max_tokens=500,
                temperature=0.2,
                # Reasoning is ON by default for this model family and must be disabled
                # via the chat template (a system-prompt phrase does not work).
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            self._log('plan', model, t0, error=repr(e))
            raise
        message = response.choices[0].message
        if not message.tool_calls:
            self._log('plan', model, t0, response, tool=None)
            print(f"[model_client DEBUG] no tool call. finish_reason={response.choices[0].finish_reason}")
            print(f"[model_client DEBUG] reasoning={getattr(message, 'reasoning', None)}")
            print(f"[model_client DEBUG] content={message.content!r}")
            return None, None, response
        call = message.tool_calls[0]
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError:
            args = {}
        self._log('plan', model, t0, response, tool=call.function.name)
        return call.function.name, args, response

    # ------------------------------------------------------------------- vision
    def chat_with_image(self, model: str, system_prompt: str, user_text: str, image_b64: str) -> str:
        """One image + one question -> the model's text answer (OpenAI image_url data URI)."""
        t0 = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    },
                ],
                max_tokens=300,
                temperature=0.2,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as e:
            self._log('vision', model, t0, error=repr(e))
            raise
        self._log('vision', model, t0, response)
        return response.choices[0].message.content or ""