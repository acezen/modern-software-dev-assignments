import os
from dotenv import load_dotenv
from ollama import chat

load_dotenv()

NUM_RUNS_TIMES = 5

# TODO: Fill this in!
YOUR_SYSTEM_PROMPT = """
你是一个精确的字符处理专家。
执行反转单词任务时，请严格遵守以下逻辑步骤：
1. 将输入单词拆分为单个字母。
2. 倒序排列这些字母。
3. 将字母重新组合。

只输出最终组合后的单词，不要输出中间过程。

<example>
input: container
letters: c o n t a i n e r
reverse: r e n i a t n o c
output: reniatnoc
</example>
<example>
input: hello
letters: h e l l o
reverse: o l l e h
output: olleh
</example>
<example>
input: http
letters: h t t p
reverse: p t t h
output: ptth
</example>
<example>
input: status
letters: s t a t u s
reverse: s u t a t s
output: sutats
</example>
"""

USER_PROMPT = """
Reverse the order of letters in the following word. Only output the reversed word, no other text:

httpstatus
"""


EXPECTED_OUTPUT = "sutatsptth"

def test_your_prompt(system_prompt: str) -> bool:
    """Run the prompt up to NUM_RUNS_TIMES and return True if any output matches EXPECTED_OUTPUT.

    Prints "SUCCESS" when a match is found.
    """
    for idx in range(NUM_RUNS_TIMES):
        print(f"Running test {idx + 1} of {NUM_RUNS_TIMES}")
        response = chat(
            model="mistral-nemo:12b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PROMPT},
            ],
            options={"temperature": 0.5},
        )
        output_text = response.message.content.strip()
        if output_text.strip() == EXPECTED_OUTPUT.strip():
            print("SUCCESS")
            return True
        else:
            print(f"Expected output: {EXPECTED_OUTPUT}")
            print(f"Actual output: {output_text}")
    return False

if __name__ == "__main__":
    test_your_prompt(YOUR_SYSTEM_PROMPT)