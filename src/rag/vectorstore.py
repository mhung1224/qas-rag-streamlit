from typing import List, Dict, Any, Optional, cast
from .config import PineconeConfig
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from .embedder import Embedder


class PineconeStore:
    def __init__(self, token: Optional[str] = None) -> None:
        self.cfg = PineconeConfig.from_env()
        self.cfg.api_key = self.cfg.api_key or token
        if not self.cfg.api_key:
            raise "PINECONE_API_KEY IS MISSING"
        self.pc = Pinecone(api_key=self.cfg.api_key)

        # Create index if it does not exist
        existing = {i["name"] for i in self.pc.list_indexes()}
        if self.cfg.index_name not in existing:
            self.pc.create_index(
                name=self.cfg.index_name,
                dimension=self.cfg.dimension,
                metric=self.cfg.metric,
                spec=ServerlessSpec(cloud=self.cfg.cloud, region=self.cfg.region),
            )
        self.index = self.pc.Index(self.cfg.index_name)

        ns = self.cfg.namespace or "default"
        try:
            stats = self.index.describe_index_stats()
        except NotFoundException:
            raise
        if ns not in stats.get("namespaces", {}):
            self.index.upsert(
                vectors=[
                    {
                        "id": "1",
                        "values": [1.0] * self.cfg.dimension,
                        "metadata": {"dummy": True}
                    }
                ],
                namespace=ns
            )

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        embeddings: List[List[float]],
    ) -> None:
        try:
            self.index.describe_index_stats()
        except NotFoundException:
            return
        
        ns = self.cfg.namespace
        bs = max(1, self.cfg.batch_size)

        def build(i: int):
            meta = metadatas[i] if (metadatas and i < len(metadatas) and isinstance(metadatas[i], dict)) else {}
            metadata = {"text": documents[i], **meta}
            return {"id": ids[i], "values": embeddings[i], "metadata": metadata}

        vectors = [build(i) for i in range(len(ids))]

        for start in range(0, len(vectors), bs):
            self.index.upsert(vectors=vectors[start:start + bs], namespace=ns)

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any] | None]:
        ns = self.cfg.namespace or "default"

        try:
            stats = self.index.describe_index_stats()
        except NotFoundException:
            return []
        if ns not in stats.get("namespaces", {}):
            return []

        res = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=ns,
            filter= cast(Dict[str, Any], {"is_dummy": {"$ne": True}})
        )

        out: List[Dict[str, Any]] = []
        for m in res.matches or []:
            md = m.metadata or {}
            out.append({
                # "id": m.id,
                "text": md.get("text", ""),
                "metadata": {k: v for k, v in md.items() if k != "text"},
            })
        return out

    def reset(self) -> None:
        try:
            stats = self.index.describe_index_stats()
        except NotFoundException:
            return
        ns = self.cfg.namespace
        if ns not in stats.get("namespaces", {}):
            return
        self.index.delete(delete_all=True, namespace=ns)

    def delete_by_source(self, source: str) -> None:
        try:
            stats = self.index.describe_index_stats()
        except NotFoundException:
            return

        ns = self.cfg.namespace or "default"
        if ns not in stats.get("namespaces", {}):
            return

        self.index.delete(filter={"source": source}, namespace=ns)

    def add_texts(self, embedder: Embedder, items: List[Dict[str, Any]]):
        if not items:
            return
        ids = [d["id"] for d in items]
        docs = [d["text"] for d in items]
        metadata = [d.get("metadata", {}) for d in items]
        embeddings = embedder.embed(docs)
        self.add(ids=ids, documents=docs, metadatas=metadata, embeddings=embeddings)

    def similarity_search(self, embedder: Embedder, query: str, top_k: int = 5):
        q_emb = embedder.embed([query], is_query = True)
        return self.query(query_embedding=q_emb, top_k=top_k)