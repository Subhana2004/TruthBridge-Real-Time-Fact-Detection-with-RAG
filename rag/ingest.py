import json
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from rag.vector_store import add_documents


SEED_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "seed_documents.json"
)


def main():
    print("Starting ingestion...")

    with open(SEED_PATH, "r", encoding="utf-8") as f:
        seed = json.load(f)

    print(f"Found domains: {list(seed.keys())}")

    for collection_name, docs in seed.items():
        print(f"Ingesting {collection_name}: {len(docs)} documents")

        add_documents(collection_name, docs)

        print(
            f"Ingested {len(docs)} documents "
            f"into collection '{collection_name}'"
        )


if __name__ == "__main__":
    main()