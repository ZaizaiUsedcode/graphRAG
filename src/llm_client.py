import httpx
from openai import OpenAI

class OnlineLLM:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )
        self.model_name = model_name

    def chat(self, prompt, temperature=0.1):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content


class OnlineEmbedding:
    def __init__(self, api_key, base_url, model_name):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )
        self.model_name = model_name

    def embed(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), 10):  # 百炼单次上限10条，分批
            batch = texts[i:i+10]
            resp = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                dimensions=1024,
                encoding_format="float"
            )
            all_embeddings.extend([d.embedding for d in resp.data])
        return all_embeddings
