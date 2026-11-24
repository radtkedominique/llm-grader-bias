import os
import boto3
from dotenv import load_dotenv

load_dotenv()
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = os.environ["AWS_BEDROCK"]

REGION = "eu-north-1"
PROVIDERS = ("openai", "meta", "amazon", "google", "deepseek")  # Nicht-Anthropic

bedrock = boto3.client("bedrock", region_name=REGION)
runtime = boto3.client("bedrock-runtime", region_name=REGION)

ids = set()
for m in bedrock.list_foundation_models(byOutputModality="TEXT")["modelSummaries"]:
    ids.add(m["modelId"])
for p in bedrock.list_inference_profiles()["inferenceProfileSummaries"]:
    ids.add(p["inferenceProfileId"])

ids = sorted(i for i in ids if any(p in i for p in PROVIDERS))

for model_id in ids:
    try:
        runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
            inferenceConfig={"maxTokens": 5},
        )
        status = "OK"
    except Exception as e:
        msg = str(e)
        if "service control policy" in msg:
            status = "GESPERRT (SCP)"
        elif "AccessDenied" in msg:
            status = "kein Zugriff (Model access / IAM)"
        elif "on-demand throughput" in msg:
            status = "nur via Inference-Profil"
        elif "ValidationException" in msg:
            status = "erlaubt, aber andere API nötig?"
        else:
            status = "FEHLER: " + msg[:80]
    print(f"{status:35} {model_id}")