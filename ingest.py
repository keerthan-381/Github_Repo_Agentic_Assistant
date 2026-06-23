from langchain_chroma import Chroma

from langchain_community.embeddings import FastEmbedEmbeddings

from document_loader import (
    load_repository_documents
)


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DB_PATH = "./chroma_db"


def create_vector_store(
    repo_path
):

    documents = (
        load_repository_documents(
            repo_path
        )
    )
    print(
        "Documents loaded:",
        len(documents)
        )
    if not documents:
        raise ValueError(
            "No documents were loaded from repository."
            )

    embeddings = (
        FastEmbedEmbeddings(
            model_name=
            EMBEDDING_MODEL
        )
    )

    try:

        existing_db = Chroma(
            persist_directory=
            DB_PATH,

            embedding_function=
            embeddings
        )

        ids = existing_db.get().get(
            "ids",
            []
        )

        if ids:

            existing_db.delete(
                ids=ids
            )

    except:
        pass
    if not documents:
        raise ValueError(
            "No documents were loaded from repository."
            )

    vectordb = (
        Chroma.from_documents(
            documents=
            documents,

            embedding=
            embeddings,

            persist_directory=
            DB_PATH
        )
    )

    print(
        f"\nVector DB Ready"
    )

    print(
        f"Total Chunks Indexed: "
        f"{len(documents)}"
    )

    return (
        vectordb,
        documents
    )
