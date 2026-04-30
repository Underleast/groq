from fastapi import FastAPI, Request
import httpx
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

LITELLM_URL = os.getenv(
    "LITELLM_URL",
    "https://groq-zt83.onrender.com/v1/chat/completions"
)

# -------- Anthropic -> OpenAI --------
def to_openai(data):
    return {
        "model": "groq",
        "messages": data.get("messages", []),
        "temperature": data.get("temperature", 1),
        "max_tokens": data.get("max_tokens", 1024),
    }

# -------- OpenAI -> Anthropic --------
def to_anthropic(data):
    # Check if LiteLLM returned an error
    if "error" in data:
        logger.error(f"LiteLLM error: {data['error']}")
        text = f"[Error from backend: {data['error'].get('message', 'Unknown error')}]"
    else:
        try:
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to extract text from response: {e}, data: {data}")
            text = ""

    return {
        "id": "msg_" + data.get("id", "local"),
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": text
            }
        ],
        "model": data.get("model", "claude-3-opus-20240229"),
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0)
        }
    }

# -------- MAIN ENDPOINT --------
@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()

    # sanitize unsupported params
    body.pop("reasoning_effort", None)
    body.pop("tools", None)
    body.pop("browser_search", None)

    openai_payload = to_openai(body)

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(LITELLM_URL, json=openai_payload)
        data = res.json()
        logger.info(f"LiteLLM response: {data}")

    anthropic_response = to_anthropic(data)
    logger.info(f"Anthropic response: {anthropic_response}")
    from fastapi.responses import JSONResponse
    return JSONResponse(content=anthropic_response)

# health check
from fastapi import Response

@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"status": "ok"}
