from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from datasets import load_dataset
from PIL import Image
import base64
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from chromadb.utils.data_loaders import ImageLoader


st.title("Flower Arrangement Query and Image Retrieval Service")


@st.cache_data
def load_flower_dataset():
    return load_dataset("huggan/flowers-102-categories")


ds = load_flower_dataset()
chroma_client = chromadb.PersistentClient(path="./data/flower.db")
image_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()
flower_collection = chroma_client.get_or_create_collection(
    "flowers_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)


def show_image_from_uri(uri, width=200):
    img = Image.open(uri)
    st.image(img, width=width)


def format_prompt_inputs(data, user_query):
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

    return inputs


def query_db(query, results=2):
    res = flower_collection.query(
        query_texts=[query], n_results=results, include=["uris", "distances"]
    )
    return res


@st.cache_resource
def get_vision_model():
    return ChatOpenAI(model="gpt-4o", temperature=0.0)


vision_model = get_vision_model()
parser = StrOutputParser()

image_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a talented florist. Answer using the given image context with direct references to parts of the images provided. "
            "Use a conversational tone, and apply markdown formatting where necessary.",
        ),
        (
            "user",
            [
                {
                    "type": "text",  # Text query as one modality
                    "text": "what are some good ideas for a bouquet arrangement {user_query}",
                },
                {
                    "type": "image_url",  # First image as the second modality
                    "image_url": "data:image/jpeg;base64,{image_data_1}",
                },
                {
                    "type": "image_url",  # Second image as another modality
                    "image_url": "data:image/jpeg;base64,{image_data_2}",
                },
            ],
        ),
    ]
)

vision_chain = image_prompt | vision_model | parser

query = st.text_input("Enter your query (e.g., 'pink flower with yellow center'):")

if query:
    st.write(f"Your query: {query}")

    with st.spinner("Retrieving images..."):
        results = query_db(query)

    st.write("Here are some images based on your query:")
    for uri in results["uris"][0]:
        show_image_from_uri(uri)

    with st.spinner("Generating suggestions..."):
        prompt_input = format_prompt_inputs(results, query)
        response = vision_chain.invoke(prompt_input)

    st.markdown(f"### Suggestions for bouquet arrangement:")
    st.write(response)
