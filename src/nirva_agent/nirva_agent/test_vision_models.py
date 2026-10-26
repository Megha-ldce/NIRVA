#!/usr/bin/env python3
"""test_vision_models.py - which of your Nebius Token Factory models accept an IMAGE?

  export NEBIUS_API_KEY=...        # already set if the robot works
  python3 test_vision_models.py            # tests every model id containing 'nvidia'
  python3 test_vision_models.py some/model-id other/model-id

Sends a small image of a solid RED square on a white background and asks for its colour.
A model that answers "red" can see. A 4xx error means the model is text-only (or the id is wrong).
"""
import base64
import os
import sys

import cv2
import numpy as np
from openai import OpenAI

BASE_URL = 'https://api.tokenfactory.nebius.com/v1'
key = os.environ.get('NEBIUS_API_KEY')
if not key:
    sys.exit('NEBIUS_API_KEY is not set')
client = OpenAI(api_key=key, base_url=BASE_URL, timeout=60.0, max_retries=0)

img = np.full((160, 160, 3), 255, np.uint8)
cv2.rectangle(img, (40, 40), (120, 120), (0, 0, 255), -1)          # BGR red
ok, buf = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buf.tobytes()).decode()

models = sys.argv[1:] or sorted(m.id for m in client.models.list().data if 'nvidia' in m.id.lower())
print(f'testing {len(models)} model(s)\n')
for m in models:
    try:
        r = client.chat.completions.create(
            model=m, max_tokens=40, temperature=0,
            messages=[{'role': 'user', 'content': [
                {'type': 'text', 'text': 'What colour is the square in this image? Answer with one word.'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}]}],
            extra_body={'chat_template_kwargs': {'enable_thinking': False}})
        ans = (r.choices[0].message.content or '').strip().replace('\n', ' ')
        verdict = 'SEES IMAGES' if 'red' in ans.lower() else 'answered but did not say red'
        print(f'{m}\n   -> {verdict}: {ans[:80]!r}')
    except Exception as e:
        print(f'{m}\n   -> no ({str(e)[:110]})')