import os
from dotenv import load_dotenv

load_dotenv()

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


# query = "pink flower with yellow center"
# results = query_db(query)
# print_results(results)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import base64

vision_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
)

parser = StrOutputParser()

image_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a talented florist and you have been asked to create a bouquet of flowers for a special event. Answer the user's question  using the given image context with direct references to parts of the images provided."
            " Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure.",
        ),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "what are some good ideas for a bouquet arrangement {user_query}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,{image_data_2}",
                },
            ],
        ),
    ]
)

vision_chain = image_prompt | vision_model | parser


def format_prompt_inputs(data, user_query):
    print("Formatting prompt inputs...")
    inputs = {}

    inputs["user_query"] = user_query

    image_path_1 = data["uris"][0][0]
    image_path_2 = data["uris"][0][1]

    with open(image_path_1, "rb") as image_file:
        image_data_1 = image_file.read()
    inputs["image_data_1"] = base64.b64encode(image_data_1).decode("utf-8")

    with open(image_path_2, "rb") as image_file:
        image_data_2 = image_file.read()
    inputs["image_data_2"] = base64.b64encode(image_data_2).decode("utf-8")

    print("Prompt inputs formatted....")
    return inputs


print("Welcome to the flower arrangement service!")
print("Please enter your query to get some ideas for a bouquet arrangement.")

query = input("Enter your query: \n")

results = query_db(query, results=2)
prompt_input = format_prompt_inputs(results, query)
response = vision_chain.invoke(prompt_input)

print("\n ------- \n")

print("\n ---Response---- \n")
print(response)

print("\n Here are some ideas for a bouquet arrangement based on your query: \n")
show_image_from_uri(results["uris"][0][0])
show_image_from_uri(results["uris"][0][1])

print("\n Images URI: \n")
print(f"Image 1: {results["uris"][0][0]}")
print(f"Image 2: {results["uris"][0][1]}")
