"""Index the bundled game records. Safe to run again without duplicate IDs."""
import os
from pathlib import Path
import json

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent


def index_games(storage_path: str, api_key: str, api_base: str | None = None) -> int:
    client = chromadb.PersistentClient(path=str(Path(storage_path) / "chromadb"))
    embedding = embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, api_base=api_base)
    collection = client.get_or_create_collection("udaplay", embedding_function=embedding)
    existing = set(collection.get(include=[])["ids"])
    for path in sorted((ROOT / "games").glob("*.json")):
        if path.stem in existing:
            continue
        game = json.loads(path.read_text(encoding="utf-8"))
        content = f"[{game['Platform']}] {game['Name']} ({game['YearOfRelease']}) - {game['Description']}"
        collection.add(ids=[path.stem], documents=[content], metadatas=[game])
    return collection.count()


if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env first.")
    storage = Path(os.getenv("UDAPLAY_DATA_DIR", ROOT / ".data")).resolve()
    count = index_games(str(storage), api_key, os.getenv("OPENAI_BASE_URL") or None)
    print(f"Indexed {count} game records in {storage / 'chromadb'}")
