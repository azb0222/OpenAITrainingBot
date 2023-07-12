import weaviate 

client = weaviate.Client("http://localhost:8080")

client = weaviate.Client(
    url = "https://some-endpoint.weaviate.network",  # Replace with your endpoint
    auth_client_secret=weaviate.AuthApiKey(api_key="YOUR-WEAVIATE-API-KEY"),  # Replace w/ your Weaviate instance API key
    additional_headers = {
        "X-HuggingFace-Api-Key": "YOUR-HUGGINGFACE-API-KEY"  # Replace with your inference API key
    }
)


class_obj = {
  "class": "Document", 
       "moduleConfig": {
               "text2vec-transformers": {
                    "skip": False,
                    "vectorizeClassName": False,
                    "vectorizePropertyName": False
                },
                "generative-openai": {
                    "model": "text-davinci-003"
                }
           },
       "vectorIndexType": "hnsw",
       "vectorizer": "text2vec-transformers",
       "properties": [
         {
           "name": "title", 
           "dataType": ["text"], 
           "moduleConfig": { 
             "text2vec-transformers": { 
                "skip": False, 
                "vectorizePropertyName": False, 
                
               }
            }
         }, 
         {
           "name": "body", 
           "dataType": ["text"], 
           "moduleConfig": { 
             "text2vec-transformers": { 
                "skip": False, 
                "vectorizePropertyName": False, 
                
               }
            }
         }, 
         {
           "name": "category", 
           "dataType": ["Category"], #need to add 
           "description": "The category this clip is associated with.",
           "moduleConfig": { 
             "text2vec-transformers": { 
                "skip": False, 
                "vectorizePropertyName": False, 
                
               }
            }
         }
       ]
}



client.schema.create_class(class_obj)



#batch imports: import data in large quantities instead of individually, improved performance 