import re
import traceback
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
    try:
        response = client.chat.completions.create(
            model='tinyllama:latest',
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict spell and grammar correction engine.\n"
                        "Your task is to correct English sentences for spelling and grammar ONLY.\n"
                        "⚠️ IMPORTANT RULES:\n"
                        "- DO NOT include any explanation, label, or comment.\n"
                        "- DO NOT write anything before or after the corrected sentence.\n"
                        "- DO NOT add quotes or prefixes like 'Corrected sentence:' or similar.\n"
                        "- DO NOT change meaning or add new words.\n"
                        "- Only return the corrected sentence. No punctuation if it wasn’t in the original.\n"
                        "\n"
                        "Here are some examples to follow:\n"
                        "\n"
                        "Input: Whatt is the benfits of soler energee in remote ares?\n"
                        "Output: What is the benefit of solar energy in remote areas?\n"
                        "\n"
                        "Input: Haw dose klimate chang afect agriculturel lands?\n"
                        "Output: How does climate change affect agricultural lands?\n"
                        "\n"
                        "Input: I hav no ideal how to fixx this issue.\n"
                        "Output: I have no idea how to fix this issue.\n"
                        "\n"
                        "Input: This sentense hass too manny errorrs.\n"
                        "Output: This sentence has too many errors.\n"
                        "\n"
                        "Now, correct the next sentence. Follow the same format strictly."
                    )
                },
                {
                    "role": "user",
                    "content": f"{text.strip()}"
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return text


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
