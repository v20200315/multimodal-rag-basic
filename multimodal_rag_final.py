import os

from PIL import Image
from matplotlib import pyplot as plt
from datasets import load_dataset

ds = load_dataset("huggan/flowers-102-categories")

# print(ds.num_rows)
# flower = ds["train"][78]["image"]
# plt.imshow(flower)
# plt.axis("off")
# plt.show()


def show_image_from_uri(uri):
    img = Image.open(uri)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


dataset_folder = "./dataset/flowers-102-categories"
os.makedirs(dataset_folder, exist_ok=True)


def save_images(dataset, dataset_folder, num_images=500):
    for i in range(num_images):
        print(f"Saving image {i+1} of {num_images}")
        image = dataset["train"][i]["image"]
        image.save(os.path.join(dataset_folder, f"flower_{i+1}.png"))
    print(f"Saved the first 500 images to {dataset_folder}")


# save_images(ds, dataset_folder, num_images=500)

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

chroma_client = chromadb.PersistentClient(path="./data/flower.db")
image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()
flower_collection = chroma_client.get_or_create_collection(
    "flowers_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

ids = []
uris = []

for i, filename in enumerate(sorted(os.listdir(dataset_folder))):
    if filename.endswith(".png"):
        file_path = os.path.join(dataset_folder, filename)

        ids.append(str(i))
        uris.append(file_path)

# flower_collection.add(ids=ids, uris=uris)
# print("Images added to the database.")
# print(flower_collection.count())


def query_db(query, results=5):
    print(f"Querying the database for: {query}")
    results = flower_collection.query(
        query_texts=[query], n_results=results, include=["uris", "distances"]
    )
    return results


def print_results(results):
    for idx, uri in enumerate(results["uris"][0]):
        print(f"ID: {results['ids'][0][idx]}")
        print(f"Distance: {results['distances'][0][idx]}")
        print(f"Path: {uri}")
        show_image_from_uri(uri)
        print("\n")


query = "pink flower with yellow center"
results = query_db(query)
print_results(results)
