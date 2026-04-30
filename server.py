from fastapi import FastAPI, Request
import httpx
import os

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
    try:
        text = data["choices"][0]["message"]["content"]
    except:
        text = ""

    return {
        "content": [
            {
                "type": "text",
                "text": text
            }
        ]
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

    return to_anthropic(data)

# health check
@app.get("/")
def health():
    return {"status": "ok"}
