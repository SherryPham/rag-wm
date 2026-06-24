from pathlib import Path


def test_package_imports():
    import sentinelrag
    from sentinelrag.rag import VectorStore
    from sentinelrag.utils.paths import default_chroma_dir, repo_root

    assert sentinelrag.__version__
    assert VectorStore is not None
    assert repo_root().name == "sentinelrag"
    assert default_chroma_dir() == Path(repo_root()) / "chromadb_db"

