import weaviate 

client = weaviate.Client("http://localhost:8080")

client = weaviate.Client(
    url = "https://some-endpoint.weaviate.network",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="YOUR-WEAVIATE-API-KEY"),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-HuggingFace-Api-Key": "YOUR-HUGGINGFACE-API-KEY"  # Replace with your inference API key
    }
)

# define class 
class_obj = {
    "class": "Question",
    #vectorizer module: transforms raw text into numerical vectors, algos need numebers
    "vectorizer": "text2vec-huggingface",  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
    "moduleConfig": {
        "text2vec-huggingface": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",  # Can be any public or private Hugging Face model.
            "options": {
                "waitForModel": True
            }
        }
    }
}

client.schema.create_class(class_obj)



#batch imports: import data in large quantities instead of individually, improved performance 