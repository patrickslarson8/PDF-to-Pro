import json
import chromadb

class VectorDB:
    """
    Database to store and retrieve context before inference.

    Attributes:
        client: Database interaction object.
        collection: Database object.
    """
    def __init__(self, name, chunk_path):
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(name=name)
        self._fill_db(self._get_chunks(chunk_path))

    def _fill_db(self, chunks: list[str]):
        """
        Idempotent function to add all context objects.
        :param chunks: Sections of text to insert as context.
        :return: None
        """
        for chunk in chunks:
            self.collection.upsert(
                documents=[chunk["text"]],
                metadatas=[{
                    # "section_hierarchy": "/".join(chunk["section_hierarchy"]),
                    # "page_start": chunk["page_start"],
                    # "page_end": chunk["page_end"],
                    "chunk_id": chunk["chunk_id"]
                }],
                ids=[chunk["chunk_id"]]
            )

    def _get_chunks(self, chunk_path: str) -> list[str]:
        """
        Retrieves context stored as json objects.
        :param chunk_path: Location of chunks.
        :return: chunks parsed into list.
        """
        chunks = []
        with open(chunk_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def retrieve_relevant_chunks(self, query: str, n:int=3) -> list[str]:
        """
        Finds and returns the n-most alike contexts based on the query.
        :param query: Query to search.
        :param n: Number of contexts to return.
        :return: List of contexts.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n
        )

        contexts = []
        for i, doc in enumerate(results["documents"][0]):
            # metadata = results["metadatas"][0][i]
            context = f"\n\n{doc}"
            contexts.append(context)

        return contexts