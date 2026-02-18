For **truly free ongoing usage** (no exhaustion surprises), switch to one of these reliable free-tier providers below.

### Recommended Free Alternatives (Copy-Paste Ready for Your Script)

| Provider          | Base URL (paste exactly)                              | Recommended Free Model                  | Free Tier (Feb 2026)                  | Get Key Here |
|-------------------|-------------------------------------------------------|-----------------------------------------|---------------------------------------|--------------|
| **Groq** (fastest) | `https://api.groq.com/openai/v1`                     | `llama-3.3-70b-versatile`              | Thousands of requests/day             | console.groq.com |
| **Google Gemini** (best quality) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.5-flash` or `gemini-2.5-pro` | Very generous (hundreds–thousands/day) | ai.google.dev |
| **OpenRouter** (most free models) | `https://openrouter.ai/api/v1`                       | `google/gemini-2.5-flash:free`<br>or `meta-llama/llama-3.3-70b-instruct:free` | Dozens of completely free models      | openrouter.ai/keys |
| **Cerebras**      | `https://api.cerebras.ai/v1`                          | `llama-3.1-70b`                        | 1 million tokens per day              | cloud.cerebras.ai |
| **Mistral**       | `https://api.mistral.ai/v1`                           | `mistral-small-latest`                 | Generous experiment tier              | console.mistral.ai |

### How to Switch in Your Script (`ai3.py`)
1. Type `exit` at the "You:" prompt to quit CLI mode.
2. Re-run the script: `python ai3.py`
3. When it asks for **API Base URL**, paste one from the table above.
4. Enter the matching **Model name**.
5. Paste your new free API key.
6. Choose CLI mode again.

**Example for Groq (super fast & reliable):**
- Base URL: `https://api.groq.com/openai/v1`
- Model: `llama-3.3-70b-versatile`
- Key: from console.groq.com (free, instant)

