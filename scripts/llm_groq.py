import os
from groq import Groq


class GroqLLM:
    def __init__(self, model: str = "llama-3.1-8b-instant", max_tokens: int = 700, temperature: float = 0.3):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def invoke(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content