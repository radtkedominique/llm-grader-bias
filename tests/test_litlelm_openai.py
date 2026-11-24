"""Call an LLM through the HAI proxy.

The proxy is OpenAI-compatible, so the normal ``openai`` client works. Only the base URL changes.

Run:
    export HAI_PROXY_KEY=<your API key>
    python hai_proxy_example.py
"""

import os

import openai
from dotenv import load_dotenv

load_dotenv()

PROXY_BASE_URL = "https://litellm.dev.haufe.ai"
LLM_MODEL = "gpt-6-luna"

client = openai.OpenAI(api_key=os.environ["HAI_PROXY_KEY"], base_url=PROXY_BASE_URL)

# 1. Plain text answer
completion = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "system", "content": "You answer German tax questions in one sentence."},
        {"role": "user", "content": "Was ist die Kleinunternehmerregelung?"},
    ],
)
print(completion.choices[0].message.content)

