import requests
import json

LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "llama-3.1-8b-instruct"
TIMEOUT = 600

# --- Simple tools (replace with real ones later) ---
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

# --- Streaming with realtime tokens + common sampling params ---
def stream_model(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS_SCHEMAS,
        "tool_choice": "auto",
        "stream": True,
        # Common sampling parameters
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0
    }

    print("\nAssistant: ", end="", flush=True)

    accumulated_content = ""
    accumulated_tool_calls = []
    accumulated_function_call = {"name": "", "arguments": ""}

    try:
        with requests.post(LLAMA_SERVER_URL, json=payload, stream=True, timeout=TIMEOUT) as resp:
            resp.raise_for_status()

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

                # Realtime token output
                if delta.get("content") is not None:
                    token = delta["content"]
                    print(token, end="", flush=True)
                    accumulated_content += token

                # Accumulate tool calls (parallel support)
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

                # Legacy single function_call
                if "function_call" in delta:
                    fc = delta["function_call"]
                    if "name" in fc:
                        accumulated_function_call["name"] += fc.get("name", "")
                    if "arguments" in fc:
                        accumulated_function_call["arguments"] += fc.get("arguments", "")

            print()  # Newline after stream ends

    except Exception as e:
        print(f"\n[Error: {e}]")
        accumulated_content += f"\n[Error: {e}]"

    # Build final assistant message
    msg = {"role": "assistant", "content": accumulated_content.strip() or None}

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
            "content": "You are a helpful assistant. Use tools when needed. After tool results, always give a clear final answer."
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
                break  # Final answer

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
                    print(f"   ← Result received\n")
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
                print(f"   ← Result received\n")
                tool_messages.append({
                    "role": "tool",
                    "name": fname,
                    "content": json.dumps(result)
                })

            history.extend(tool_messages)

        print("-" * 60)


if __name__ == "__main__":
    chat()