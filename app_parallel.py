from concurrent.futures import ThreadPoolExecutor
import traceback
import re
from openai import OpenAI

# Initialize Ollama-compatible client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ---------- Utility Functions ----------

def clean_corrected_output(text: str) -> str:
    cleaned_text = re.sub(
        r"^(Corrected Sentence:|The corrected sentence is:|Can you summarize.*?:)",
        "",
        text,
        flags=re.IGNORECASE
    ).strip()
    match = re.search(r"([A-Za-z][^.?!]*[.?!])", cleaned_text)
    return match.group(1).strip() if match else cleaned_text

def is_math_expression(text: str) -> bool:
    return bool(re.fullmatch(r"[\d\s\+\-\*\/\.\(\)]+", text.strip()))

# ---------- Tool A: Keyword Extractor ----------

def extract_keywords(text):
    return [word.strip(".,!?") for word in text.split() if len(word) > 4]

# ---------- Tool B: Mock Web Search ----------

def mock_web_search(keywords):
    return f"Search results for: {', '.join(keywords)}. Example content about {keywords[0]} and its importance."

# ---------- Tool C: Summarizer ----------

def summarize_with_qwen(text):
    completion = client.chat.completions.create(
        model="qwen3:4b",
        messages=[
            {"role": "system", "content": "You are a helpful summarizer. Focus only on the core ideas and ignore instruction-like prefixes."},
            {"role": "user", "content": f"Summarize this:\n\n{text}"}
        ]
    )
    return completion.choices[0].message.content

# ---------- Tool D: Calculator ----------

def evaluate_math_expression(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

# ---------- Tool E: Spell Checker (TinyLLM) ----------

def spell_check(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model='tinyllama:latest',
            messages=[
                {"role": "system", "content": "You are a spell correction engine. ONLY return the corrected sentence in plain English. Do NOT explain or add extra text."},
                {"role": "user", "content": f"Correct this sentence:\n\n{text}"}
            ]
        )
        raw_output = response.choices[0].message.content
        return clean_corrected_output(raw_output)
    except Exception as e:
        traceback.print_exc()
        return text  # fallback: return original text

# ---------- Thread-safe Tool Wrappers ----------

def safe_extract_keywords(user_input):
    try:
        return extract_keywords(user_input)
    except Exception:
        traceback.print_exc()
        return []

def safe_mock_search(keywords):
    try:
        return mock_web_search(keywords)
    except Exception:
        traceback.print_exc()
        return "Search failed."

def safe_summarize(text):
    try:
        return summarize_with_qwen(text)
    except Exception:
        traceback.print_exc()
        return "Summary generation failed."

def safe_calculate(expression):
    try:
        return evaluate_math_expression(expression)
    except Exception:
        traceback.print_exc()
        return "Invalid math expression."

# ---------- Main Orchestrator ----------

def multi_tool_assistant(user_input):
    print("User Input:", user_input)

    if is_math_expression(user_input):
        print("🧮 Detected math input. Routing to calculator...")
        result = safe_calculate(user_input)
        print("Calculator Result:\n", result)
        return

    # Step 0: Spell check
    print("Running spell check...")
    corrected_input = spell_check(user_input)
    print("Corrected Input:", corrected_input)

    results = {}

    with ThreadPoolExecutor() as executor:
        future_keywords = executor.submit(safe_extract_keywords, corrected_input)
        keywords = future_keywords.result()

        futures = {
            'search': executor.submit(safe_mock_search, keywords),
        }

        if keywords:
            search_result = futures['search'].result()
            futures['summary'] = executor.submit(safe_summarize, search_result)

        for name, future in futures.items():
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"[{name.upper()} ERROR]", e)
                traceback.print_exc()
                results[name] = f"{name} failed."

    # Final Output
    print("\nCorrected Query Used:\n", corrected_input)
    if 'summary' in results:
        print("\n🧠 Qwen Summary:\n", results['summary'])
    else:
        print("\nNo summary generated.")
    print("\n🔎 Search Result:\n", results.get('search'))


# ---------- Run Demo ----------

if __name__ == "__main__":
    queries = [
        "Whatt is the benfits of soler energee in remote ares?",
        "Haw dose klimate chang afect agriculturel lands?",
        "25*(35/7+2)"
    ]
    for q in queries:
        print("\n" + "=" * 60)
        multi_tool_assistant(q)
