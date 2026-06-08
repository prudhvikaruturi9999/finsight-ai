"""Quick sanity check — run 3 queries against the ChromaDB vector store."""
from embedding.models.local_model import LocalEmbeddingModel
from vector_store.chroma_store import ChromaStore

model = LocalEmbeddingModel()
store = ChromaStore()
print(f"Total chunks in store: {store.count()}\n")

queries = [
    "What was Google total revenue in 2025?",
    "Operating income and operating margin",
    "Risk factors related to AI and competition",
]

for q in queries:
    vec, _ = model.embed([q])
    results = store.query(vec[0], top_k=3, filters={"chunk_type": "child"})
    print(f"Q: {q}")
    for i, r in enumerate(results, 1):
        src = r["metadata"]["source_path"].replace("\\", "/").split("/")[-1]
        pg = r["metadata"]["page_no"]
        score = r["score"]
        snippet = r["text"][:150].strip().replace("\n", " ")
        print(f"  {i}. [{src}  p.{pg}]  score={score:.3f}  {snippet!r}")
    print()
