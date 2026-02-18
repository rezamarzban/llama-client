import requests
import json
import time
import os
import importlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# ====================== GLOBAL CONFIG & HISTORY ======================
WEB_PORT = 8000
TIMEOUT = 600
MAX_RETRIES = 3

# Default values (local Llama)
API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "llama-3.1-8b-instruct"
API_KEY = ""

HISTORY = [
    {
        "role": "system",
        "content": "You are an expert helpful assistant.\n"
                   "When user asks for multiple items, call the tool multiple times in parallel.\n"
                   "For complex problems use tools sequentially if needed.\n"
                   "Always give final answer after tools."
    }
]

TOOL_INTERACTIONS = []


def update_config(new_url, new_model, new_key):
    global API_URL, MODEL, API_KEY
    API_URL = new_url.strip()
    MODEL = new_model.strip()
    API_KEY = new_key.strip()
    print(f"✅ Config updated → URL: {API_URL} | Model: {MODEL}")


# ====================== TOOL LOADER ======================
def load_tools():
    tools = {}
    schemas = []
    for filename in os.listdir('.'):
        if filename.endswith('_tool.py'):
            module_name = filename[:-3]
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, 'schema'):
                    sch = module.schema
                    tool_name = sch['function']['name']
                    if hasattr(module, tool_name):
                        tools[tool_name] = getattr(module, tool_name)
                        schemas.append(sch)
                        print(f"   Loaded tool: {tool_name}")
            except Exception as e:
                print(f"   Failed to load {filename}: {e}")
    return tools, schemas


TOOLS, TOOLS_SCHEMAS = load_tools()
print(f"✅ Loaded {len(TOOLS)} tools\n")


# ====================== STREAMING ======================
def stream_model(messages, send_func=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS_SCHEMAS,
        "tool_choice": "auto",
        "stream": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096
    }

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    conn_type = "Cloud API" if API_KEY else "Local Server"
    if send_func:
        send_func(f"data: Connecting to {conn_type}...\n\n")
    else:
        print(f"\nConnecting to {conn_type}...", end="", flush=True)

    accumulated_content = ""
    accumulated_tool_calls = []
    accumulated_function_call = {"name": "", "arguments": ""}
    token_count = 0
    start_time = None

    for attempt in range(MAX_RETRIES):
        try:
            with requests.post(API_URL, json=payload, headers=headers, stream=True, timeout=TIMEOUT) as resp:
                resp.raise_for_status()

                if send_func:
                    send_func("data: Connected ✓\n\n")
                else:
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

                    if delta.get("content") is not None:
                        token = delta["content"]
                        if start_time is None:
                            start_time = time.time()
                        accumulated_content += token
                        token_count += 1
                        if send_func:
                            send_func(f"data: {token}\n\n")
                        else:
                            print(token, end="", flush=True)

                    if "tool_calls" in delta:
                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            while len(accumulated_tool_calls) <= idx:
                                accumulated_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            tc = accumulated_tool_calls[idx]
                            if "id" in tc_delta:
                                tc["id"] += tc_delta.get("id", "")
                            if "function" in tc_delta:
                                f = tc_delta["function"]
                                if f.get("name"): tc["function"]["name"] += f.get("name", "")
                                if f.get("arguments"): tc["function"]["arguments"] += f.get("arguments", "")

                    if "function_call" in delta:
                        fc = delta["function_call"]
                        if "name" in fc:
                            accumulated_function_call["name"] += fc.get("name", "")
                        if "arguments" in fc:
                            accumulated_function_call["arguments"] += fc.get("arguments", "")

                break

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                err = f"Connection failed: {e}"
                if send_func:
                    send_func(f"data: {err}\n\n")
                else:
                    print(" Failed ✗")
                    print(f"   [{err}]")
                accumulated_content = err
                break
            wait = (2 ** attempt) * 1.5
            if send_func:
                send_func(f"data: Retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s...\n\n")
            else:
                print(f" Failed (retry {attempt+1}/{MAX_RETRIES} in {wait:.1f}s)", flush=True)
            time.sleep(wait)
            if not send_func:
                print("Connecting...", end="", flush=True)

    if token_count > 3 and start_time is not None:
        speed = token_count / (time.time() - start_time)
        speed_str = f"  ({speed:.1f} tokens/s)"
        if send_func:
            send_func(f"data: {speed_str}\n\n")
        else:
            print(speed_str, end="")

    if not send_func:
        print()

    accumulated_content = accumulated_content.strip().encode('utf-8', 'ignore').decode('utf-8')

    msg = {"role": "assistant", "content": accumulated_content or None}

    if accumulated_tool_calls:
        clean = [tc for tc in accumulated_tool_calls if tc["function"]["name"]]
        if clean:
            msg["tool_calls"] = clean
    elif accumulated_function_call["name"]:
        msg["function_call"] = accumulated_function_call

    return msg


# ====================== CONVERSATION TURN ======================
def process_turn(history, send_func=None):
    step = 0
    max_steps = 20
    while step < max_steps:
        step += 1
        assistant_msg = stream_model(history, send_func)
        history.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls")
        function_call = assistant_msg.get("function_call")

        if not tool_calls and not function_call:
            return assistant_msg.get("content", "")

        if send_func:
            send_func("data: 🔧 Tool(s) being used...\n\n")
        else:
            print("\n   🔧 Tool(s) being used...\n")

        calls = tool_calls or ([{"function": function_call}] if function_call else [])

        for tc in calls:
            fname = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except:
                args = {}

            short_report = f"🔧 Used tool: {fname}({args})"
            if send_func:
                send_func(f"data: {short_report}\n\n")
            else:
                print(f"   {short_report}")

            result = TOOLS.get(fname)(**args) if fname in TOOLS else {"error": "Unknown tool"}

            TOOL_INTERACTIONS.append({
                "tool": fname,
                "args": args,
                "result": result,
                "time": time.strftime("%H:%M:%S")
            })

        history.extend([{
            "role": "tool",
            "tool_call_id": tc.get("id", "call_1"),
            "name": tc["function"]["name"],
            "content": json.dumps(result)
        } for tc in calls])

    return ""


# ====================== TOOL TEST ======================
def test_single_tool():
    print("\n=== Tool Tester ===")
    print("Available tools:", list(TOOLS.keys()))
    name = input("Tool name (empty to cancel): ").strip()
    if not name or name not in TOOLS:
        return
    args_str = input("Arguments as JSON (empty = {}): ").strip()
    try:
        args = json.loads(args_str) if args_str else {}
    except:
        args = {}
    print(f"Calling {name}...")
    result = TOOLS[name](**args)
    print("Result:")
    print(json.dumps(result, indent=4))
    TOOL_INTERACTIONS.append({
        "tool": name,
        "args": args,
        "result": result,
        "time": time.strftime("%H:%M:%S")
    })


# ====================== HTTP HANDLER ======================
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ('/', '/index.html'):
            try:
                with open('index.html', 'rb') as f:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f.read())
            except:
                self.send_response(404)
                self.end_headers()
        elif path == '/tools':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(list(TOOLS.keys())).encode())
        elif path == '/tool-log':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(TOOL_INTERACTIONS, default=str).encode())
        elif path == '/config':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {"url": API_URL, "model": MODEL, "key": "******" if API_KEY else ""}
            self.wfile.write(json.dumps(data).encode())
        elif path.startswith('/chat'):
            prompt = parse_qs(parsed.query).get('prompt', [''])[0].strip()
            if not prompt:
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            def send(line):
                try:
                    self.wfile.write((line + "\n\n").encode('utf-8'))
                    self.wfile.flush()
                except:
                    pass

            HISTORY.append({"role": "user", "content": prompt})
            process_turn(HISTORY, send_func=send)
            send("data: [DONE]\n\n")
        elif path.startswith('/test'):
            query = parse_qs(parsed.query)
            tool = query.get('tool', [''])[0]
            args_str = query.get('args', ['{}'])[0]
            try:
                args = json.loads(args_str)
            except:
                args = {}
            if tool in TOOLS:
                result = TOOLS[tool](**args)
                TOOL_INTERACTIONS.append({
                    "tool": tool,
                    "args": args,
                    "result": result,
                    "time": time.strftime("%H:%M:%S")
                })
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/config':
            try:
                length = int(self.headers['Content-Length'])
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                update_config(data.get('url', API_URL), data.get('model', MODEL), data.get('key', API_KEY))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def run_web():
    print(f"\n🌐 Web server running at http://localhost:{WEB_PORT}")
    print("   Open browser → Config tab to change API settings anytime")
    server = HTTPServer(('', WEB_PORT), Handler)
    server.serve_forever()


# ====================== CLI MODE ======================
def run_cli():
    print("\n=== CLI Mode ===")
    print("Commands: config | test | tools | exit\n")
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

        if user.lower() == "config":
            print("\n=== Current Configuration ===")
            print(f"URL   : {API_URL}")
            print(f"Model : {MODEL}")
            print(f"Key   : {'******' if API_KEY else '(empty)'}")
            print("\nEnter new values (press Enter to keep current):")
            new_url = input(f"URL   [{API_URL}]: ").strip() or API_URL
            new_model = input(f"Model [{MODEL}]: ").strip() or MODEL
            new_key = input(f"Key   [{'******' if API_KEY else ''}]: ").strip()
            if new_key == "": new_key = API_KEY
            update_config(new_url, new_model, new_key)
            continue

        if user.lower() == "test":
            test_single_tool()
            continue
        if user.lower() == "tools":
            print("\n=== Tools Interactions Log ===")
            if not TOOL_INTERACTIONS:
                print("No tool calls yet.")
            else:
                for i, entry in enumerate(TOOL_INTERACTIONS, 1):
                    print(f"{i}. [{entry['time']}] {entry['tool']}")
                    print(f"   Args : {entry['args']}")
                    print(f"   Result: {json.dumps(entry['result'], indent=2)}")
                    print("-" * 60)
            continue

        HISTORY.append({"role": "user", "content": user})
        process_turn(HISTORY)


# ====================== START ======================
if __name__ == "__main__":
    print("Choose mode:")
    print("1. CLI mode")
    print("2. Web mode (with Config tab)")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        run_cli()
    elif mode == "2":
        run_web()
    else:
        print("Invalid choice. Exiting.")
