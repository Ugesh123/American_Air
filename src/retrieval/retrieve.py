"""
Semantic retrieval over the train corpus (25,451 customer->AmericanAir pairs).

Two backends, same interface (`Retriever.query(text, k) -> list[RetrievedCase]`):

1. TfidfRetriever (default, ACTIVE): TF-IDF + cosine similarity via scikit-learn.
   Works fully offline, laptop-friendly, no extra installs. This is what every
   number in results/ was actually computed with.

2. EmbeddingRetriever (optional, not benchmarked yet): sentence-transformers
   + FAISS behind the same interface. It needs two extra packages:
       pip install sentence-transformers faiss-cpu
   No number in results/ comes from this backend; it is kept as the planned
   next experiment (see REPORT.md, "Next steps").

What is embedded/indexed: `customer_text_clean` from data/processed/train.jsonl
(the customer's message, not the AA reply) -- we retrieve similar PAST CUSTOMER
MESSAGES and return their associated `aa_text_clean` as grounding evidence for
the new message, because we want "how did AA historically respond to a similar
complaint", not "find me a similar-sounding AA reply".

How many results are retrieved: top-5 by default (`k=5`), but only entries
above `MIN_SIMILARITY` are kept -- irrelevant low-similarity matches are
filtered out entirely rather than padding the evidence list.

Retrieval confidence: the top-1 cosine similarity score itself (0-1). This is
NOT calibrated to any external scale -- it is used only for RELATIVE decisions
(escalation policy thresholding, golden-set weak-retrieval flagging), and that
is stated explicitly wherever it's used.
"""
import json
import os
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN_PATH = os.path.join(ROOT, 'data', 'processed', 'train.jsonl')

MIN_SIMILARITY = 0.12  # below this, a "retrieved" case is treated as noise, not evidence


@dataclass
class RetrievedCase:
    pair_id: str
    customer_text: str
    aa_reply_text: str
    similarity: float


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]


class TfidfRetriever:
    def __init__(self, train_path=TRAIN_PATH):
        self.train = load_jsonl(train_path)
        texts = [r['customer_text_clean'] for r in self.train]
        self.vec = TfidfVectorizer(max_features=8000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)
        self.X = self.vec.fit_transform(texts)

    def query(self, text: str, k: int = 5) -> list[RetrievedCase]:
        qv = self.vec.transform([text])
        sims = cosine_similarity(qv, self.X)[0]
        order = sims.argsort()[::-1][:k]
        out = []
        for idx in order:
            sim = float(sims[idx])
            if sim < MIN_SIMILARITY:
                continue
            r = self.train[idx]
            out.append(RetrievedCase(
                pair_id=r['pair_id'], customer_text=r['customer_text_clean'],
                aa_reply_text=r['aa_text_clean'], similarity=sim,
            ))
        return out


class EmbeddingRetriever:
    """Optional embedding backend (sentence-transformers + FAISS) with the
    same interface as TfidfRetriever. Not used for any reported number yet.
    Requires: pip install sentence-transformers faiss-cpu
    """
    def __init__(self, train_path=TRAIN_PATH, model_name='all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError as e:
            raise ImportError(
                'sentence-transformers/faiss not installed. This backend requires '
                'internet access on your machine: pip install sentence-transformers faiss-cpu'
            ) from e
        import numpy as np
        self.train = load_jsonl(train_path)
        texts = [r['customer_text_clean'] for r in self.train]
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vecs = cosine sim
        self.index.add(embeddings.astype('float32'))
        self._np = np

    def query(self, text: str, k: int = 5) -> list[RetrievedCase]:
        qv = self.model.encode([text], normalize_embeddings=True).astype('float32')
        sims, idxs = self.index.search(qv, k)
        out = []
        for sim, idx in zip(sims[0], idxs[0]):
            if sim < MIN_SIMILARITY or idx < 0:
                continue
            r = self.train[int(idx)]
            out.append(RetrievedCase(
                pair_id=r['pair_id'], customer_text=r['customer_text_clean'],
                aa_reply_text=r['aa_text_clean'], similarity=float(sim),
            ))
        return out


def get_retriever(backend: str = 'tfidf'):
    if backend == 'tfidf':
        return TfidfRetriever()
    elif backend == 'embedding':
        return EmbeddingRetriever()
    raise ValueError(f'unknown backend: {backend}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='tfidf', choices=['tfidf', 'embedding'])
    ap.add_argument('--query', default="my flight got delayed and nobody told me anything")
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()

    retriever = get_retriever(args.backend)
    results = retriever.query(args.query, k=args.k)
    print(f'Query: {args.query!r}')
    print(f'Retrieved {len(results)} cases (min_similarity={MIN_SIMILARITY}):')
    for r in results:
        print(f'  sim={r.similarity:.3f} | customer: {r.customer_text[:80]}')
        print(f'           | AA reply:  {r.aa_reply_text[:80]}')
