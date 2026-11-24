import os
import boto3
from dotenv import load_dotenv   # pip install python-dotenv

load_dotenv()
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.environ["AWS_BEDROCK"]

client = boto3.client("bedrock-runtime", region_name="eu-north-1")

for model_id in [
    "eu.anthropic.claude-haiku-5-5",
]:
    try:
        r = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Sag kurz Hallo."}]}],
            inferenceConfig={"maxTokens": 20},
        )
        print("OK  ", model_id, "→", r["output"]["message"]["content"][0]["text"])
    except Exception as e:
        print("FAIL", model_id, "→", e)