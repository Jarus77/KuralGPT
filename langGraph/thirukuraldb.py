
import weaviate
from sentence_transformers import SentenceTransformer

class VectorDBManager:
    def _init_(self, weaviate_url="http://localhost:8080"):
        # Initialize the Weaviate client
        self.client = weaviate.Client(weaviate_url)
        print("Is client Connected")
        
        self.model = SentenceTransformer("all-distilroberta-v1")

    def create_schema(self):
        # Define the schema for the class (like table in SQL)
        schema = {
            "classes": [
                {
                    "class": "Thirukkural",  # Name of the class
                    "description": "Thirukkural couplets",
                    "properties": [
                        {"name": "verse", "dataType": ["text"],},
                        {"name": "translation", "dataType": ["text"],},
                        {"name": "explanation", "dataType": ["text"],},
                    ]
                }
            ]
        }
        
        # Create schema in Weaviate
        self.client.schema.create(schema)
        print("Schema created successfully!")


    def add_data(self, entry):
        try:
            data_object = {
                "verse": entry["kural"],
                "translation": entry["translation"],
                "explanation": entry["explanation"],
            }
            
            vector = self.model.encode([entry["kural"]+entry["explanation"]+entry["translation"]])
            vector = vector[0]
        
            # Add the object to Weaviate
            return self.client.data_object.create(
                data_object,
                vector=vector,
                class_name="Thirukkural" 
            )
            print(f"Data added: successfully")
        except Exception as e:
            print(f"error while insertion: {e}")


    def findby_vector(self,query):
        vector = self.model.encode([query])
        vector = vector[0]
        query = self.client.query.get("Thirukkural", ["verse", "translation", "explanation"]) \
           .with_near_vector({"vector": vector}) \
           .with_limit(1) \
           .do()
        data = query["data"]["Get"]["Thirukkural"]
        return data



    def query_data(self, val):
        try:
            # Execute the query with the filter
            query = """
            {
                Get {
                    Thirukkural(where: {
                        path: ["explanation"],
                        operator: Equal,
                        valueString: "%s"
                    }) {
                        chapter
                        section
                        verse
                        translation
                        explanation
                    }
                }
            }
            """ % val  # Format the query with the value
            
            # Execute the query with the filter
            result = self.client.query.raw(query)

            if result and "data" in result and result["data"]["Get"]["Thirukkural"]:
                return result["data"]["Get"]["Thirukkural"]
            else:
                return "No results found."
        except Exception as e:
            return f"Error while querying: {e}"


        

    def list_all_data(self):
        try:
            # Execute the query to fetch all objects in the Thirukkural class
            query = """
            {
                Get {
                    Thirukkural {
        
                        verse
                    
                        explanation
                    }
                }
            }
            """
            
            # Execute the query
            result = self.client.query.raw(query)

            # Check if any data was returned
            if result and "data" in result and result["data"]["Get"]["Thirukkural"]:
                return result["data"]["Get"]["Thirukkural"]
            else:
                return "No data found."
        except Exception as e:
            return f"Error while querying: {e}"


    def semantic_search(self,query):
        result = self.client.query.get("Thirukkural", ["Verse", "Explanation"]).with_near_text({
                    "concepts": ["What is the value of knowledge?"]
                }).with_limit(3).do()
        print(result)