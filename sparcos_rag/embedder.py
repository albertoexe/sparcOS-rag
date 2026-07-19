class Embedder:
    def __init__(self, model: str, host: str, client=None):
        self.model_name = model
        if client is None:
            import ollama
            client = ollama.Client(host=host)
        self._client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            resp = self._client.embeddings(model=self.model_name, prompt=t)
            out.append([float(x) for x in resp["embedding"]])
        return out
