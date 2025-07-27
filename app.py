import re
from openai import OpenAI

# Initialize Ollama Client
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


# ----------- Utility Functions -----------

def clean_corrected_output(text: str) -> str:
    """
    Removes instruction-like prefixes and extracts the first valid sentence.
    """
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


# ----------- Tool A: Keyword Extractor -----------

def extract_keywords(text: str) -> list[str]:
    return [word.strip(".,!?") for word in text.split() if len(word) > 4]


# ----------- Tool B: Mock Web Search -----------

def mock_web_search(keywords: list[str]) -> str:
    return f"Search results for: {', '.join(keywords)}. Example content about '{keywords[0]}' and its importance."


# ----------- Tool C: Qwen Summarizer -----------

def summarize_with_qwen(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen3:4b",
        messages=[
            {"role": "system", "content": "You are an expert summarizer. Provide a concise, well-structured summary that captures the key points and main ideas."},
            {"role": "user", "content": f"Summarize the following content in 2–3 concise sentences, focusing only on the main idea and ignoring any instruction-like phrases (e.g., 'Corrected Sentence:', etc.).\n\n{text}"}
        ]
    )
    return completion.choices[0].message.content


# ----------- Tool D: Calculator -----------

def evaluate_math_expression(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ----------- Tool E: Spell Checker -----------

def spell_check(text: str) -> str:
    response = client.chat.completions.create(
        model='tinyllama:latest',
        messages=[
            {"role": "system", "content": "You are a spell correction engine. ONLY return the corrected sentence in plain English. Do NOT explain or add extra text."},
            {"role": "user", "content": f"Correct this sentence:\n\n{text}"}
        ]
    )
    raw_output = response.choices[0].message.content
    return clean_corrected_output(raw_output)


# ----------- Main Orchestrator -----------

def multi_tool_assistant(user_input: str):
    print("User Input:", user_input)

    if is_math_expression(user_input):
        print("🧮 Detected math input. Routing to Calculator.")
        result = evaluate_math_expression(user_input)
        print("Tool D - Calculator:\n", result)
        return

    corrected = spell_check(user_input)
    print("Tool E - Spell Checker:\n", corrected)

    keywords = extract_keywords(corrected)
    print("Tool A - Extracted Keywords:", keywords)

    search_result = mock_web_search(keywords)
    print("Tool B - Search Result:\n", search_result)

    summary = summarize_with_qwen(search_result)
    print("Tool C - Qwen Summary:\n", summary)


# ----------- Run Demo -----------

if __name__ == "__main__":
    test_inputs = [
        "Tell me abot the impact of solar powar on rurl comunities.",
        "Whatt is the benfits of soler energee in remote ares?",
        "Discus the impct of edukation on developing countrees.",
        "Haw dose klimate chang afect agriculturel lands?",
        "25*(3+7)"
    ]

    for query in test_inputs:
        print("\n" + "=" * 60)
        multi_tool_assistant(query)
