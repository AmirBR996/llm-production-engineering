import time
from litellm import completion, completion_cost

OLLAMA_BASE = "http://127.0.0.1:11434"


def classify_task(user_query: str) -> str:
    cls = completion(
        model="ollama/qwen3:8b",
        api_base=OLLAMA_BASE,
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify the following query into EXACTLY one word: "
                    "'code', 'summary', or 'general'. "
                    f"Query: {user_query}\n\nAnswer:"
                )
            }
        ],
        max_tokens=5
    )

    return cls.choices[0].message.content.strip().lower()


def call_with_fallbacks(model_chain, messages):
    last_error = None

    for model in model_chain:
        try:
            kwargs = {
                "model": model,
                "messages": messages
            }

            if model.startswith("ollama/"):
                kwargs["api_base"] = OLLAMA_BASE

            return completion(**kwargs)

        except Exception as e:
            print(
                f" {model} failed "
                f"({type(e).__name__}), trying next..."
            )
            last_error = e

    raise last_error


def smart_chat(user_query: str):
    task = classify_task(user_query)

    routing = {
        "code": [
            "ollama/gemma4:31b-cloud",
            "groq/openai/gpt-oss-20b"
        ],
        "summary": [
            "ollama/gemma4:31b-cloud",
            "groq/openai/gpt-oss-20b"
        ],
        "general": [
            "groq/openai/gpt-oss-20b",
            "ollama/gemma4:31b-cloud"
        ]
    }

    model_chain = routing.get(task, routing["general"])

    start = time.time()

    response = call_with_fallbacks(
        model_chain=model_chain,
        messages=[
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    latency = time.time() - start

    try:
        cost = completion_cost(completion_response=response)
        cost_str = f"${cost:.6f}"
    except Exception:
        cost_str = "n/a"

    return {
        "detected_task": task,
        "model_used": response.model,
        "answer": response.choices[0].message.content,
        "latency_sec": round(latency, 2),
        "cost_usd": cost_str
    }


queries = [
    "Write a Python function to compute Fibonacci numbers.",
    "Summarize the importance of attention mechanism in 2 sentences.",
    "Tell me a fun fact about elephants."
]

for q in queries:
    print("\nQuestion:", q)

    result = smart_chat(q)

    print(f"Task:    {result['detected_task']}")
    print(f"Model:   {result['model_used']}")
    print(f"Latency: {result['latency_sec']}s")
    print(f"Cost:    {result['cost_usd']}")
    print(f"Answer:  {result['answer']}")