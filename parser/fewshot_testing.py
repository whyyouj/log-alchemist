from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

examples = [
    {
        "question": "What is the tallest mountain in the world?",
        "answer": "Mount Everest",
    },
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"question": "In which year did the first airplane fly?", "answer": "1903"},
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\n{answer}",
)
prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)

print(
    prompt_template.format(
        input="What is the name of the famous clock tower in London?"
    )
)