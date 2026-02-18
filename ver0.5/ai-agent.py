import requests
import json
import time
import os
import importlib
import uuid
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ====================== GLOBAL CONFIG & HISTORY ======================
WEB_PORT = 8000
TIMEOUT = 600
MAX_RETRIES = 3
MAX_HISTORY = 20  # Keep only the last N messages to prevent context overflow

# Default values (Compatible with Llama.cpp / Oobabooga / vLLM)
API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "llama-3.1-8b-instruct"
API_KEY = ""

# Initial System Prompt
HISTORY = [
    {
        "role": "system",
        "content": (
            "You are an expert helpful assistant.\n"
            "When a user asks for multiple items, call the tool multiple times in parallel.\n"
            "For complex problems use tools sequentially if needed.\n"
            "Always give a final answer after tools have finished."
        )
    }
]

TOOL_INTERACTIONS = []

def update_config(new_url, new_model, new_key):
    global API_URL, MODEL, API_KEY
    API_URL = new_url.strip()
    MODEL = new_model.strip()
    API_KEY = new_key.strip()
    print(f"✅ Config updated → URL: {API_URL} | Model: {MODEL}")

def trim_history():
    """Keeps the system prompt + last N messages."""
    global HISTORY
    if len(HISTORY) > MAX_HISTORY:
        # Keep system prompt (index 0) + last (MAX_HISTORY - 1) messages
        HISTORY = [HISTORY[0]] + HISTORY[-(MAX_HISTORY - 1):]

# ====================== TOOL LOADER ======================
def load_tools():
    tools = {}
    schemas = []
    # Look for *_tool.py files in the current directory
    for filename in os.listdir('.'):
        if filename.endswith('_tool.py'):
            module_name = filename[:-3]
            try:
                module = importlib.import_module(module_name)
                # Reload module to catch updates if code changed while running
                importlib.reload(module)
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

# ====================== STREAMING LOGIC ======================
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
    status_msg = f"data: Connecting to {conn_type}...\n\n"
    if send_func:
        send_func(status_msg)
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

                if send_func: send_func("data: Connected ✓\n\n")
                else: print(" Connected ✓")

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

                    if not chunk.get("choices"): continue
                    delta = chunk["choices"][0].get("delta", {})

                    # Handle Content
                    if delta.get("content"):
                        token = delta["content"]
                        if start_time is None: start_time = time.time()
                        accumulated_content += token
                        token_count += 1
                        if send_func: send_func(f"data: {token}\n\n")
                        else: print(token, end="", flush=True)

                    # Handle Tool Calls (Standard)
                    if "tool_calls" in delta:
                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            while len(accumulated_tool_calls) <= idx:
                                accumulated_tool_calls.append({
                                    "id": "", 
                                    "type": "function", 
                                    "function": {"name": "", "arguments": ""}
                                })
                            tc = accumulated_tool_calls[idx]
                            if "id" in tc_delta: tc["id"] += tc_delta.get("id", "")
                            if "function" in tc_delta:
                                f = tc_delta["function"]
                                if f.get("name"): tc["function"]["name"] += f.get("name", "")
                                if f.get("arguments"): tc["function"]["arguments"] += f.get("arguments", "")

                    # Handle Deprecated Function Call
                    if "function_call" in delta:
                        fc = delta["function_call"]
                        if "name" in fc: accumulated_function_call["name"] += fc.get("name", "")
                        if "arguments" in fc: accumulated_function_call["arguments"] += fc.get("arguments", "")

                break # Success, exit retry loop

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                err = f"Connection failed: {e}"
                if send_func: send_func(f"data: {err}\n\n")
                else: print(f" Failed ✗ [{err}]")
                accumulated_content = err
                break
            time.sleep((2 ** attempt) * 1)

    if not send_func: print()

    msg = {"role": "assistant", "content": accumulated_content or None}

    # Normalize tool calls
    if accumulated_tool_calls:
        # Filter out empty calls
        clean = [tc for tc in accumulated_tool_calls if tc["function"]["name"]]
        if clean: msg["tool_calls"] = clean
    elif accumulated_function_call["name"]:
        # Convert legacy function_call to modern tool_calls
        msg["tool_calls"] = [{
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": accumulated_function_call
        }]

    return msg

# ====================== CONVERSATION TURN ======================
def process_turn(history, send_func=None):
    trim_history()
    step = 0
    max_steps = 10 # Prevent infinite loops
    
    while step < max_steps:
        step += 1
        assistant_msg = stream_model(history, send_func)
        
        # --- ID RECOVERY LOGIC ---
        # Local models sometimes forget to generate IDs. We must ensure they exist.
        if assistant_msg.get("tool_calls"):
            for tc in assistant_msg["tool_calls"]:
                if not tc.get("id"):
                    tc["id"] = f"call_{uuid.uuid4().hex[:8]}"
        
        history.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls", [])

        if not tool_calls:
            return assistant_msg.get("content", "")

        if send_func: send_func("data: 🔧 Tool(s) being used...\n\n")
        else: print("\n   🔧 Tool(s) being used...")

        # Execute tools
        for tc in tool_calls:
            call_id = tc["id"]
            fname = tc["function"]["name"]
            args_str = tc["function"]["arguments"]
            
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                # Fallback: sometimes local models leave trailing commas
                try: 
                    import re
                    clean_args = re.sub(r',\s*([\]}])', r'\1', args_str)
                    args = json.loads(clean_args)
                except:
                    args = {"error": "JSON parse failed", "raw": args_str}

            short_report = f"🔧 {fname}({args})"
            if send_func: send_func(f"data: {short_report}\n\n")
            else: print(f"   {short_report}")

            # CALL THE TOOL
            if fname in TOOLS:
                try:
                    result = TOOLS[fname](**args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Tool '{fname}' not found"}

            TOOL_INTERACTIONS.append({
                "tool": fname, "args": args, "result": result, "time": time.strftime("%H:%M:%S")
            })

            # Append result to history
            history.append({
                "role": "tool",
                "tool_call_id": call_id,
                "name": fname,
                "content": json.dumps(result)
            })

    return ""

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
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"index.html not found")
                
        elif path == '/tools':
            self.send_json(list(TOOLS.keys()))
            
        elif path == '/tool-log':
            self.send_json(TOOL_INTERACTIONS)
            
        elif path == '/config':
            data = {"url": API_URL, "model": MODEL, "key": "******" if API_KEY else ""}
            self.send_json(data)
            
        elif path.startswith('/chat'):
            query = parse_qs(parsed.query)
            prompt = query.get('prompt', [''])[0].strip()
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            # Robust send function that handles client disconnects
            def send(line):
                try:
                    self.wfile.write((line + "\n\n").encode('utf-8'))
                    self.wfile.flush()
                except (ConnectionResetError, BrokenPipeError):
                    pass # Client disconnected, just stop writing

            if prompt:
                HISTORY.append({"role": "user", "content": prompt})
                process_turn(HISTORY, send_func=send)
            
            send("data: [DONE]\n\n")

        elif path.startswith('/test'):
            query = parse_qs(parsed.query)
            tool = query.get('tool', [''])[0]
            args_str = query.get('args', ['{}'])[0]
            try: args = json.loads(args_str)
            except: args = {}
            
            if tool in TOOLS:
                result = TOOLS[tool](**args)
                TOOL_INTERACTIONS.append({
                    "tool": tool, "args": args, "result": result, "time": time.strftime("%H:%M:%S")
                })
                self.send_json(result)
            else:
                self.send_error(404, "Tool not found")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/config':
            try:
                length = int(self.headers['Content-Length'])
                data = json.loads(self.rfile.read(length).decode('utf-8'))
                update_config(data.get('url', API_URL), data.get('model', MODEL), data.get('key', API_KEY))
                self.send_json({"status": "ok"})
            except:
                self.send_error(400)
        else:
            self.send_error(404)

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())

    def log_message(self, format, *args):
        pass # Suppress default logging

def run_web():
    print(f"\n🌐 Web server running at http://localhost:{WEB_PORT}")
    print("   Open browser → Config tab to change API settings anytime")
    # ThreadingHTTPServer allows multiple requests (e.g., config checks while chatting)
    server = ThreadingHTTPServer(('', WEB_PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

# ====================== CLI MODE ======================
def run_cli():
    print("\n=== CLI Mode ===")
    while True:
        try: user = input("You: ").strip()
        except: break
        if user.lower() in ("exit", "quit"): break
        if not user: continue
        
        HISTORY.append({"role": "user", "content": user})
        process_turn(HISTORY)

# ====================== START ======================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        run_cli()
    else:
        run_web()
