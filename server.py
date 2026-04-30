from fastapi import FastAPI, Request, Response
import httpx
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"

# -------- Anthropic -> OpenAI --------
def to_openai(data):
    return {
        "model": GROQ_MODEL,
        "messages": data.get("messages", []),
        "temperature": data.get("temperature", 0.7),
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
    body.pop("thinking", None)
    body.pop("response_format", None)

    openai_payload = to_openai(body)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(GROQ_URL, json=openai_payload, headers=headers)
        data = res.json()
        logger.info(f"Groq response status: {res.status_code}")

    anthropic_response = to_anthropic(data)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=anthropic_response)

# health check
@app.api_route("/", methods=["GET", "HEAD"])
@app.api_route("/health", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return {"status": "ok", "groq_key_set": bool(GROQ_API_KEY)}
