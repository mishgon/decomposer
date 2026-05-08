DECOMPOSER_SYSTEM_PROMPT = (
    "You are Decomposer. You write concise Python function bodies that answer "
    "multi-hop questions by composing calls to an available single-hop QA "
    "function."
)


DECOMPOSER_FEW_SHOT_PROMPT = '''Here are examples:

def main(context, answer_single_hop_question):
    """Answer the question: When was the institute that owned The Collegian founded?

    Args:
        context (str): A context that contains the necessary information to
            answer a question.
        answer_single_hop_question (Callable[[str, str], str]): Function that
            takes two arguments:
            - question (str): A single-hop question answerable from context.
            - context (str): The same formatted MuSiQue context string passed
              to main.
            It returns a short answer string grounded in the context.

    Returns:
        str: The final answer to the question.
    """
    answer_1 = answer_single_hop_question('Which institute owned The Collegian?', context)
    answer_2 = answer_single_hop_question(f'When was {answer_1} founded?', context)
    return answer_2

---

def main(context, answer_single_hop_question):
    """Answer the question: Who started the Bethel branch of the religion founded by the black community in the city that used to be the US capitol?

    Args:
        context (str): A context that contains the necessary information to
            answer a question.
        answer_single_hop_question (Callable[[str, str], str]): Function that
            takes two arguments:
            - question (str): A single-hop question answerable from context.
            - context (str): The same formatted MuSiQue context string passed
              to main.
            It returns a short answer string grounded in the context.

    Returns:
        str: The final answer to the question.
    """
    answer_1 = answer_single_hop_question('What religion did the black community found?', context)
    answer_2 = answer_single_hop_question('Which city used to be the US capitol?', context)
    answer_3 = answer_single_hop_question(f'Who started the Bethel branch of {answer_1} in {answer_2}?', context)
    return answer_3

---

def main(context, answer_single_hop_question):
    """Answer the question: Where is the lowest place in the country which, along with Eisenhower's VP's country, recognized Gaddafi's government early on?

    Args:
        context (str): A context that contains the necessary information to
            answer a question.
        answer_single_hop_question (Callable[[str, str], str]): Function that
            takes two arguments:
            - question (str): A single-hop question answerable from context.
            - context (str): The same formatted MuSiQue context string passed
              to main.
            It returns a short answer string grounded in the context.

    Returns:
        str: The final answer to the question.
    """
    answer_1 = answer_single_hop_question("Who served as Eisenhower's vice president?", context)
    answer_2 = answer_single_hop_question(f'What country was {answer_1} president of?', context)
    answer_3 = answer_single_hop_question(f"Along with {answer_2}, what major power recognized Gaddafi's government at an early date?", context)
    answer_4 = answer_single_hop_question(f'Where is the lowest place in {answer_3}?', context)
    return answer_4'''


def build_decomposer_messages(func_spec: str) -> list[dict]:
    user_content = f"""Generate the output function body for the input function spec.

Use only Python built-ins, `context`, and `answer_single_hop_question`.
Do not define helper functions.
Do not output Markdown fences.
Output only the output function body, indented with four spaces.
The prompt already ends with the four spaces for the first body line, so your
first output line must start directly with Python code.
All later top-level body lines must start with four spaces.

{DECOMPOSER_FEW_SHOT_PROMPT}

---

{func_spec}
""" + "    "
    return [
        {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


SINGLE_HOP_QA_SYSTEM_PROMPT = (
    "You answer single-hop questions using only the provided context. "
    "Return only the short answer string. Do not explain."
)


SINGLE_HOP_QA_FEW_SHOT_PROMPT = '''Examples:

Context:
[0] The Collegian (Houston Baptist University)
The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas.

Question: Which institute owned The Collegian?
Answer: Houston Baptist University

---

Context:
[0] Green (Steve Hillage album)
Green is the fourth studio album by British progressive rock musician Steve Hillage.

Question: Who is the performer of Green?
Answer: Steve Hillage'''


def build_single_hop_qa_messages(question: str, context: str) -> list[dict]:
    user_content = f"""{SINGLE_HOP_QA_FEW_SHOT_PROMPT}

---

Context:
{context}

Question: {question}
Answer:"""
    return [
        {"role": "system", "content": SINGLE_HOP_QA_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


BASELINE_FEW_SHOT_PROMPT = '''Examples:

Context:
[0] The Collegian (Houston Baptist University)
The Collegian is the bi-weekly official student publication of Houston Baptist University in Houston, Texas.

[1] Houston Baptist University
Houston Baptist University is a private Baptist university in Sharpstown, Houston, Texas. It was founded in 1960.

Question: When was the institute that owned The Collegian founded?
Answer: 1960

---

Context:
[0] Green (Steve Hillage album)
Green is the fourth studio album by British progressive rock musician Steve Hillage.

[1] Miquette Giraudy
Miquette Giraudy is a keyboard player and vocalist, best known for her work in Gong and with her partner Steve Hillage.

Question: Who is the spouse of the Green performer?
Answer: Miquette Giraudy'''


def build_baseline_messages(question: str, context: str) -> list[dict]:
    user_content = f"""{BASELINE_FEW_SHOT_PROMPT}

---

Context:
{context}

Question: {question}
Answer: """
    return [
        {"role": "user", "content": user_content},
    ]
