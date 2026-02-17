import requests
import json
import time

LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "llama-3.1-8b-instruct"
TIMEOUT = 600
MAX_RETRIES = 3

# --- Simple tools ---
def search_web(query):
    return {"results": f"Fake search results for '{query}'"}

def graphic_art(prompt):
    return {"image_url": f"https://example.com/{prompt.replace(' ', '_')}.png"}

TOOLS = {"search_web": search_web, "graphic_art": graphic_art}

# --- Tool schemas ---
TOOLS_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "graphic_art",
            "description": "Generate graphic art based on a prompt.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string", "description": "The prompt describing the desired graphic art."}},
                "required": ["prompt"]
            }
        }
    }
]

# --- Streaming with realtime connection status + accurate token speed ---
def stream_model(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS_SCHEMAS,
        "tool_choice": "auto",
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }

    print("\nConnecting to server...", end="", flush=True)

    accumulated_content = ""
    accumulated_tool_calls = []
    accumulated_function_call = {"name": "", "arguments": ""}
    token_count = 0
    start_time = None   # Will be set when FIRST token arrives

    for attempt in range(MAX_RETRIES):
        try:
            with requests.post(LLAMA_SERVER_URL, json=payload, stream=True, timeout=TIMEOUT) as resp:
                resp.raise_for_status()
                print(" Connected ✓", flush=True)
                print("Assistant: ", end="", flush=True)

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    # Realtime tokens
                    if delta.get("content") is not None:
                        token = delta["content"]
                        # Start timer on FIRST token
                        if start_time is None:
                            start_time = time.time()
                        print(token, end="", flush=True)
                        accumulated_content += token
                        token_count += 1

                    # Tool calls accumulation
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            while len(accumulated_tool_calls) <= idx:
                                accumulated_tool_calls.append({
                                    "id": "", "type": "function", "function": {"name": "", "arguments": ""}
                                })
                            tc = accumulated_tool_calls[idx]
                            if "id" in tc_delta:
                                tc["id"] += tc_delta.get("id", "")
                            if "function" in tc_delta:
                                f = tc_delta["function"]
                                if "name" in f:
                                    tc["function"]["name"] += f.get("name", "")
                                if "arguments" in f:
                                    tc["function"]["arguments"] += f.get("arguments", "")

                    if "function_call" in delta:
                        fc = delta["function_call"]
                        if "name" in fc:
                            accumulated_function_call["name"] += fc.get("name", "")
                        if "arguments" in fc:
                            accumulated_function_call["arguments"] += fc.get("arguments", "")

                break  # Success

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(" Failed ✗")
                print(f"   [Connection error: {e}]")
                accumulated_content = f"[Error: {e}]"
                break
            wait = (2 ** attempt) * 1.5
            print(f" Failed (retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s)", flush=True)
            time.sleep(wait)
            print("Connecting to server...", end="", flush=True)

    # Accurate tokens/s (only from first token)
    if token_count > 3 and start_time is not None:
        elapsed = time.time() - start_time
        speed = token_count / elapsed
        print(f"  ({speed:.1f} tokens/s)", end="")
    print()  # final newline

    # Clean weird characters
    accumulated_content = accumulated_content.strip().replace("ð", "").replace("�", "")

    # Build message
    msg = {"role": "assistant", "content": accumulated_content or None}

    if accumulated_tool_calls:
        clean = []
        for tc in accumulated_tool_calls:
            if tc["function"]["name"]:
                clean.append({
                    "id": tc["id"] or None,
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                })
        if clean:
            msg["tool_calls"] = clean

    elif accumulated_function_call["name"]:
        msg["function_call"] = accumulated_function_call

    return msg


# --- Main chat ---
def chat():
    history = [
        {
            "role": "system",
            "content": "You are a helpful assistant. If the user asks to search for several things, call search_web multiple times in parallel using several tool_calls at once."
        }
    ]

    print("Assistant ready — type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})

        step = 0
        max_steps = 10

        while step < max_steps:
            step += 1
            assistant_msg = stream_model(history)
            history.append(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls")
            function_call = assistant_msg.get("function_call")

            if not tool_calls and not function_call:
                break

            # Tool execution
            print("   🔧 Calling tool(s)...")
            tool_messages = []

            if tool_calls:
                for tc in tool_calls:
                    fname = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except:
                        args = {}
                    print(f"   → {fname}({args})")
                    result = TOOLS.get(fname)(**args) if fname in TOOLS else {"error": "Unknown tool"}
                    print(f"   📤 Tool output: {result}\n")
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", "call_1"),
                        "name": fname,
                        "content": json.dumps(result)
                    })

            elif function_call:
                fname = function_call["name"]
                try:
                    args = json.loads(function_call["arguments"])
                except:
                    args = {}
                print(f"   → {fname}({args})")
                result = TOOLS.get(fname)(**args) if fname in TOOLS else {"error": "Unknown tool"}
                print(f"   📤 Tool output: {result}\n")
                tool_messages.append({
                    "role": "tool",
                    "name": fname,
                    "content": json.dumps(result)
                })

            history.extend(tool_messages)

        print("-" * 60)


if __name__ == "__main__":
    chat()
