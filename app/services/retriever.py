# M21 — RAG Foundation: Document Store & Retrieval
#
# ---------------------------------------------------------------------------
# PRODUCTION NOTES (M21)
#
# Vector database:
#   Replace the in-memory FAISS index with a proper vector DB (Qdrant, Weaviate,
#   pgvector, Pinecone). The in-memory index is rebuilt from scratch on every
#   restart, does not persist between deploys, and cannot be shared across
#   multiple containers. A vector DB solves all three.
#
# Precomputed embeddings:
#   Documents in the KB should be embedded once and stored. Only the query
#   embedding is computed at runtime. Currently every Retriever() instantiation
#   re-embeds all 33 KB entries — expensive if the process restarts frequently.
#
# Hybrid search:
#   Dense vector search handles semantic similarity well but underperforms on
#   exact product names, version numbers, and error codes. Combine with BM25
#   keyword search (reciprocal rank fusion is the standard merge strategy) for
#   better recall on those cases. Most vector DBs support hybrid search natively.
#
# Separate service:
#   The retriever should be independently deployable so it can be scaled,
#   its index updated, and its latency monitored without touching the LLM steps.
#   Retrieval latency already has its own span in M27 — use that to detect
#   vector DB slowdowns independently of LLM slowdowns.
# ---------------------------------------------------------------------------
#
# READING ORDER:
#   1. KnowledgeEntry  — typed representation of one knowledge base record
#   2. Retriever       — loads the KB, builds the FAISS index, exposes retrieve()
#   3. _embed()        — calls Ollama's embedding API; implemented for you
#
# WHY FAISS + COSINE SIMILARITY?
#   FAISS is a library for efficient similarity search over dense vectors.
#   We embed each resolution into a fixed-size vector (768 dimensions for
#   nomic-embed-text) and store all of them in an index at startup.
#   At query time we embed the incoming ticket and ask FAISS for the K
#   vectors most similar to it — this is much faster than re-calling the
#   embedding model for every stored document on every query.
#
#   Cosine similarity measures the angle between vectors, not their magnitude.
#   A perfect match = 1.0, completely unrelated = 0.0.
#   FAISS's IndexFlatIP (inner product) gives cosine similarity when vectors
#   are L2-normalised first — normalising is the trick that converts inner
#   product into cosine similarity without needing a dedicated cosine index.
#
# WHY EMBED RESOLUTIONS, NOT TICKETS?
#   At query time we have a new ticket but no resolution yet. We retrieve the
#   most relevant *past resolutions* so the reply generator can ground its
#   answer in real solutions. Embedding the resolutions means we match
#   "what was done" rather than "what was said" — more useful for generation.

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import faiss

from app.config import settings


@dataclass
class KnowledgeEntry:
    ticket: str
    category: str
    resolution: str


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self):
        # Retrieving training data from a JSON file is a common pattern for small-scale RAG implementations.
        with open(settings.knowledge_base_path, encoding="utf-8") as f:
            data = json.load(f)
        #Normalizing the data into KnowledgeEntry dataclass.
        self.entries = [KnowledgeEntry(**row) for row in data]
        # Applying embeding and building the FAISS index at startup for efficient retrieval.
        embedded_entries = [self._embed(entry.resolution) for entry in self.entries]
        # Transforming into a matrix and normalizing for cosine similarity, then building the FAISS index.
        # This is important for efficient retrieval: without an index, we'd have to compute similarity against every entry on every query.

        # np.vstack() was used here instead of np.array() because we have a list of 2D arrays (each embedding is shape (1, D)) 
        # and we want to stack them into a single 2D array of shape (N, D) where N is the number of entries. 
        # np.array() would create an array of shape (N, 1, D) which is not what we want for FAISS.
        matrix = np.vstack(embedded_entries).astype(np.float32)
        faiss.normalize_L2(matrix)
        D = matrix.shape[1]
        # Faiss's IndexFlatIP is used instead of IndexFlatL2 because we normalise the vectors first, which makes inner product equivalent to cosine similarity.
        # IndexFlatL2 would compute Euclidean distance, which is not what we want for measuring semantic similarity in this context. 
        # Because we want to retrieve the most semantically similar resolutions, cosine similarity is the appropriate choice, and normalising the vectors allows us to use the more efficient inner product index for this purpose.
        # Meanwhile IndexFlatIp computes the inner product between the query vector and the stored vectors, which, after normalisation, effectively gives us the cosine similarity scores needed for retrieval.
        # Effectively retrieving the most relevant past resolutions based on the semantic content of the ticket, which is crucial for grounding the reply generation in real solutions.
        self._index = faiss.IndexFlatIP(D)
        self._index.add(matrix)

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        # scores is shape (1, k) — cosine similarities descending.
        # indices is shape (1, k) — positions in self.entries.
        vec = self._embed(query)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, k)
        return [self.entries[i].resolution for i in indices[0]]

    def retrieve_with_scores(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Same as retrieve() but also returns the cosine similarity score for each hit.

        Scores are in [0, 1] — 1.0 is identical, ~0.70+ is a strong match,
        below ~0.50 is weak. Use this in experiments to observe score drop-off
        on out-of-domain queries; pipeline.py uses retrieve() which discards scores.
        """
        vec = self._embed(query)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, k)
        return [
            (self.entries[indices[0][j]].resolution, float(scores[0][j]))
            for j in range(len(indices[0]))
        ]

    # -------------------------------------------------------------------------
    # Internal — embedding call
    #
    # Calls Ollama's /api/embed endpoint directly via httpx.
    # Returns a normalised float32 numpy array of shape (1, D).
    # This is implemented for you — the interesting work is in __init__ and retrieve().
    # -------------------------------------------------------------------------

    # Embedding is the process of converting text into a fixed-size vector representation that captures its semantic meaning. 
    # This allows us to compare the similarity of different pieces of text based on their vector representations, which is crucial for our retrieval system.
    def _embed(self, text: str) -> np.ndarray:
        response = httpx.post(
            f"{settings.ollama_api_base}/api/embed",
            json={"model": settings.embed_model, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        vector = response.json()["embeddings"][0]
        return np.array(vector, dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# FeatureRetriever (M28)
#
# A second FAISS index over data/feature_kb.json — answers "what features
# does this product actually have?" so the classifier can ground wrong-premise
# decisions in evidence rather than Signature instructions.
#
# WHY A SEPARATE CLASS INSTEAD OF REUSING RETRIEVER:
#   Retriever embeds `resolution` (what was done for a past ticket).
#   FeatureRetriever embeds `description` (what a feature does and where to
#   find it). The embedded field, the output format, and the KB file are all
#   different. A shared base class would be a premature abstraction at this
#   scale — two explicit classes make the parallel pattern visible.
#
# WHY EMBED DESCRIPTION, NOT FEATURE NAME:
#   A ticket saying "I can't believe there's no dark mode" will not match
#   the feature name "Dark Mode" well. It matches the description — "can be
#   enabled from Settings > Appearance > Theme" — because the description
#   contains the context a confused user would be searching for.
#
# retrieve() OUTPUT FORMAT:
#   Returns formatted strings ready to be joined and passed as feature_context
#   to ClassifyTicket. Format: "Feature: {name}\n{description}"
#   The label "Feature:" gives the model a parse anchor so it knows the
#   following text describes a real product capability, not a user complaint.
# ---------------------------------------------------------------------------

@dataclass
class FeatureEntry:
    feature: str
    description: str
    category: str


class FeatureRetriever:
    # Path is derived from knowledge_base_path so no new config field is needed.
    _KB_PATH = Path(settings.knowledge_base_path).parent / "feature_kb.json"

    def __init__(self):
        feature_data = json.loads(self._KB_PATH.read_text(encoding="utf-8"))
        self.entries = [FeatureEntry(**entry) for entry in feature_data]
        embedded_entries = [self._embed(entry.description) for entry in self.entries]
        matrix = np.vstack(embedded_entries).astype(np.float32)
        faiss.normalize_L2(matrix)
        D = matrix.shape[1]
        self._index = faiss.IndexFlatIP(D)
        self._index.add(matrix)

    def retrieve(self, query: str, k: int = 2) -> list[str]:
        # k=2: the classifier rarely needs more than two features to resolve a
        # wrong-premise. More context risks diluting the relevant signal.
        vec = self._embed(query)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, k)
        return [
            f"Feature: {self.entries[i].feature}\n{self.entries[i].description}"
            for i in indices[0]
        ]

    def _embed(self, text: str) -> np.ndarray:
        # Identical to Retriever._embed() — shared embedding model, same API.
        # In production this would be extracted to a module-level function
        # to avoid duplication, but keeping it explicit here makes the
        # two-class parallel structure easier to read.
        response = httpx.post(
            f"{settings.ollama_api_base}/api/embed",
            json={"model": settings.embed_model, "input": text},
            timeout=30.0,
        )
        response.raise_for_status()
        vector = response.json()["embeddings"][0]
        return np.array(vector, dtype=np.float32).reshape(1, -1)
