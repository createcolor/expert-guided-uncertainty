import sys
import json
from time import sleep
import pandas as pd
import mistralai 
from tqdm import tqdm
import httpx


# Here is the question:
# {question}
# Please provide your best answer guess (choosing from "yes", "no", "maybe") and your confidence in it from 0% (absolutely uncertain) to 100% (absolutely certain) 
# in the following JSON format:
# {{
# "answer": "Your answer here",
# "confidence_score": number
# }}

# ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
# Here is the question:
# {question}
# Please provide your best answer guess (choosing from "yes", "no", "maybe" --- a single word), 
# your confidence in it from 0% (absolutely uncertain) to 100% (absolutely certain) 
# and how difficult the question feels to you from 0 (obvious answer) to 10 (really hard, do not know the answer)
# in the following JSON format:
# {{
# "answer": "Your answer here",
# "confidence_score": number,
# "difficulty": number
# }}.
# # """.strip()

# ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
# Here is the question:
# {question}
# Please provide your best answer guess, choosing from:
# 0 --- certainly "no",
# 1 --- I am unsure, but suppose the answer is "no",
# 2 --- I am really unsure, but the answer "no" is more likely,
# 3 --- the provided context is not enough to answer the question, "yes" and "no" answers seem equally possible,
# 4 --- I am really unsure, but the answer "yes" is more likely,
# 5 --- I am unsure, but suppose the answer is "yes",
# 6 --- certainly "yes"
# in the following JSON format:
# {{
# "answer": number
# }}.
# """.strip()


# ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
# Here is the question:
# {question}
# Please provide your best answer guess, choosing from:
# 0 --- certainly "no",
# 1 --- I am unsure, but suppose the answer is "no",
# 2 --- I am really unsure, but the answer "no" is more likely,
# 3 --- the provided context is not enough to answer the question, "yes" and "no" answers seem equally possible,
# 4 --- I am really unsure, but the answer "yes" is more likely,
# 5 --- I am unsure, but suppose the answer is "yes",
# 6 --- certainly "yes",
# 7 --- the answer can be both "yes" and "no" depending on the situation (the question is ambiguous, two right answers).

# Plese provide only one number --- the answer, no text.
# """.strip()

# ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
# Here is the sentence:
# {question}
# Please provide one synonymous sentence to it (reformulate it).
# It must be a question. Answer with the single sentence, no comments.
# """.strip()

ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
Here is the sentence:
{question}
Please provide ten different synonymous sentence to it (reformulations).
All the sentences must be questions and must be different. Provide only synonymous sentences separated by "\n", no comments.
""".strip()


def apply_template(text: str) -> str:
    """
    Applies confidence template to the question.
    """
    return ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE.format(question=text)


def get_chat_response(client, model, context, question):
    try:
        chat_response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "" if context is None else context,
                    },
                    {
                        "role": "user",
                        "content": apply_template(question),
                    },
                ],
                max_tokens=300, # 1024
                temperature = 1.5 # 1.5
                # n = 10
            )
    
    except mistralai.models.sdkerror.SDKError:
        chat_response = get_chat_response(client, model, context, question)
    
    except httpx.ConnectError:
        chat_response = get_chat_response(client, model, context, question)
    
    return chat_response
    

def call_mistral(api_key: str, question: str, context: str | None = None):
    """
    Calls MistalAI API with a specified API key.
    """
    model = "mistral-medium-2505"
    client = mistralai.Mistral(api_key=api_key)

    chat_response = get_chat_response(client, model, context, question)

    print("<BEGIN RESPONSE>")
    print(chat_response.choices[0].message.content)
    print("<END RESPONSE>")
    # sleep(1)


def main():
    data = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/"
                           "pqa_labeled/train-00000-of-00001.parquet")
    data['contexts'] = data['context'].apply(lambda x: ' '.join(x['contexts']))

    for repeat in range(10):
        print(f"\n\n#### Repeat {repeat} ####\n\n")
        with open('mistral-synonyms.json', 'r', encoding='utf-8') as f:
            confidences = json.load(f)

        for row in tqdm(data.itertuples(), desc="Generating responses"):
            if str(row.pubid) in confidences.keys():
                continue
            print(row.pubid)
            call_mistral(sys.argv[1], row.question, row.contexts)
            sleep(1)
        # data.apply(lambda x: call_mistral(sys.argv[1], x['question'], x['contexts']), axis=1)

if __name__ == "__main__":
    main()
