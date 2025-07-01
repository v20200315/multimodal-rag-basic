import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt

chroma_client = chromadb.PersistentClient(path="./data/chroma.db")

image_loader = ImageLoader()

embedding_function = OpenCLIPEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

collection.add(
    ids=[
        "E23",
        "E25",
        "E33",
    ],
    uris=[
        "./images/E23-2.jpg",
        "./images/E25-2.jpg",
        "./images/E33-2.jpg",
    ],
    metadatas=[
        {
            "item_id": "E23",
            "category": "food",
            "item_name": "Braised Fried Tofu with Greens",
        },
        {
            "item_id": "E25",
            "category": "food",
            "item_name": "Sauteed Assorted Vegetables",
        },
        {"item_id": "E33", "category": "food", "item_name": "Kung Pao Tofu"},
    ],
)


def print_query_results(query_list: list, query_results: dict) -> None:
    result_count = len(query_results["ids"][0])

    for i in range(len(query_list)):
        print(f"Results for query: {query_list[i]}")

        for j in range(result_count):
            id = query_results["ids"][i][j]
            distance = query_results["distances"][i][j]
            data = query_results["data"][i][j]
            document = query_results["documents"][i][j]
            metadata = query_results["metadatas"][i][j]
            uri = query_results["uris"][i][j]

            print(
                f"id: {id}, distance: {distance}, metadata: {metadata}, document: {document}"
            )

            print(f"data: {uri}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()


query_texts = ["food with carrots", "tiger"]

query_results = collection.query(
    query_texts=query_texts,
    n_results=2,
    include=["documents", "distances", "metadatas", "data", "uris"],
)

print_query_results(query_texts, query_results)
