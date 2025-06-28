import requests
import chromadb
from sentence_transformers import SentenceTransformer
import json
import uuid
from typing import List, Dict, Any

def load_json_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class CustomEmbeddingFunction (chromadb.EmbeddingFunction):
    def __init__(self, model):
        self.model = model
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        return self.model.encode(texts)

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.model.encode([query])

class CustomModel:
    def __init__(self, model_name: str):
        """Initialize the custom model with a given name."""
        self.model_name = model_name

    def encode(self, texts: List[str]) -> List[List[float]]:
        embedingList = []
        for text in texts:
            embedding = self.embed(text)
            if not isinstance(embedding, list):
                raise ValueError(f"Embedding for text '{text}' is not a list: {embedding}")
            embedingList.append(embedding)
        return embedingList
        


    def embed(self, text: str):
        url = "http://127.0.0.1:1234/v1/embeddings"
        print("Text size : ", len(text))
        payload = json.dumps({
            "model": self.model_name,
            "input": text[:2300]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        #sleep thread for 10 seconds
        #import time
        #time.sleep(10)

        if response.status_code != 200:
            raise Exception(f"Error fetching embedding: {response.status_code} - {response.text} :: textSize : {len(text)} :: Request payload: {payload}")
        response_data = response.json()
        if 'data' not in response_data or len(response_data['data']) == 0:
            raise Exception("Invalid response format from embedding API")
        response = response_data.get("data")[0]
        
        if not isinstance(response, dict):
            raise ValueError(f"Expected response to be a dictionary, got: {type(response)}")
        
        if 'embedding' not in response:
            raise Exception("Embedding not found in response data")
        embadding = response.get("embedding")
        if not isinstance(embadding, list):
            raise ValueError(f"Expected embedding to be a list, got: {type(response.get('embedding'))}")
        print(f"Embedding for text '{text[:50]}...' generated successfully.")
        return embadding
    
class VectorDBDemo:
    def __init__(self):
        # Initialize ChromaDB client (persistent storage)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_lm_studio")
        
        # Initialize sentence transformer model (open source)
        print("Loading sentence transformer model...")
        self.embedding_model = CustomModel("text-embedding-mxbai-embed-large-v1")
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="api_data_collection",
            metadata={"description": "Collection for storing API data with embeddings"},
            embedding_function=CustomEmbeddingFunction(self.embedding_model)
        )
        
        # Display current database status
        count = self.collection.count()
        print(f"📊 ChromaDB initialized. Current documents: {count}")
        if count > 0:
            print("🔄 Database contains existing data - duplicates will be skipped")
        
    def fetch_api_data(self, api_url: str) -> List[Dict]:
        """
        Fetch JSON array from REST API
        """
        try:
            print(f"Fetching data from: {api_url}")
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Ensure we have a list
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("API response is not a JSON array or object")
                
            print(f"Successfully fetched {len(data)} records")
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching API data: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []
    
    def prepare_documents(self, data: List[Dict]) -> tuple:
        """
        Prepare documents for vector storage
        Returns: (documents, metadatas, ids)
        """
        documents = []
        metadatas = []
        ids = []
        
        for item in data:
            print(f"Item : {item}")
            print(type(item))
            # Create a text representation for embedding
            # Customize this based on your data structure
            if isinstance(item, dict):
                print("inside if")
                # Create deterministic ID based on content (prevents duplicates)
                # Use a unique field like 'id' if available, otherwise create hash
                if 'id' in item:
                    doc_id = f"doc_{item['id']}"
                else:
                    # Create hash-based ID for consistent identification
                    import hashlib
                    content_str = json.dumps(item, sort_keys=True)
                    doc_id = hashlib.md5(content_str.encode()).hexdigest()
                
                # Combine relevant text fields for embedding
                text_parts = []
                
                # Common fields to check for text content
                text_fields = ['title', 'content']
                
                for field in text_fields:
                    if field in item and item[field]:
                        text_parts.append(str(item[field]))
                
                # If no text fields found, use string representation
                if not text_parts:
                    text_parts = [str(item)]
                
                document_text = " ".join(text_parts)
                documents.append(document_text)
                
                # Store original data as metadata
                metadatas.append(item)
                
                # Use deterministic ID
                ids.append(doc_id)
        
        return documents, metadatas, ids
    
    def store_in_vector_db(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Store documents in vector database with embeddings (prevents duplicates)
        """
        if not documents:
            print("No documents to store")
            return
        
        # Check existing data count
        existing_count = self.collection.count()
        print(f"Current documents in database: {existing_count}")
        
        # Filter out documents that already exist
        new_documents = []
        new_metadatas = []
        new_ids = []
        
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            # Check if document already exists by querying
            try:
                existing = self.collection.get(ids=[doc_id])
                if existing['ids']:  # Document exists
                    print(f"⚠️  Skipping duplicate document ID: {doc_id}")
                    continue
            except:
                pass  # Document doesn't exist, we can add it
            
            new_documents.append(doc)
            #new_documents.append(self.getDocumentSummary(doc))
            new_metadatas.append(meta)
            new_ids.append(doc_id)
        
        if not new_documents:
            print("🔄 No new documents to add (all already exist)")
            return
        
        print(f"Adding {len(new_documents)} new documents...")
        print("Generating embeddings...")
        
        # Generate embeddings using sentence transformer
        #embeddings = self.embedding_model.encode(new_documents).tolist()
        batch_size = 2  # Adjust based on your system's memory
        for i in range(0, len(new_documents), batch_size):
            batch_docs = new_documents[i:i + batch_size]
            batch_metadatas = new_metadatas[i:i + batch_size]
            batch_ids = new_ids[i:i + batch_size]
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
        # Add to ChromaDB collection
        
        
        print(f"✅ Successfully added {len(new_documents)} new documents!")
        print(f"📊 Total documents in database: {self.collection.count()}")
    
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar documents using vector similarity
        """
        print(f"Searching for: '{query}'")
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results
    def display_search_results(self, results: Dict):
        """
        Display search results in a readable format
        """
        if not results['documents'] or not results['documents'][0]:
            print("No results found")
            return
        
        print("\n🔍 Search Results:")
        print("-" * 50)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"\nResult {i+1} (similarity: {1-distance:.3f}):")
            print(f"Document: {doc[:200]}...")
            print(f"Metadata: {json.dumps(metadata, indent=2)[:300]}...")
            print("-" * 30)

def callGenAI(results: Dict, query):
        """
        Display search results in a readable format
        """
        if not results['documents'] or not results['documents'][0]:
            print("No results found")
            return
        
        print("\n🔍 Search Results:")
        print("-" * 50)
        
        prompt = "You are a helpful assistant. Based on the following documents, answer query with attached documentId as reference:\n\n"
        prompt += f"Query: {query}\n\n"
        prompt += "Here are the relevant documents:\n"
        for i, (doc, metadata, distancem, ids) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0],
            results['ids'][0]
        )):
            prompt += f"\nStory id = {ids} with story content {json.dumps(metadata, indent=2)}\n"
        
        url = "http://localhost:1234/v1/chat/completions"

        payload = json.dumps({
        "model": "deepseek-r1-distill-llama-8b",
        "messages": [
            {
            "role": "system",
            "content": "Answer given prompt based on the provided documents. If you don't know the answer, just say 'I don't know'."
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
        "temperature": 0.6,
        "max_tokens": -1,
        "stream": False
        })
        
        headers = {
        'Content-Type': 'application/json'
        }

        print(f"Calling GenAI with prompt: {prompt}")

        response = requests.request("POST", url, headers=headers, data=payload)

        print(f"Response : {response.text}")

import requests
            
def fetch_task_details_byId(taskId) :
    url = f"https://frontdoor-prod-ap-southeast-2-2.clickup.com/task-v3/experience/1854334/tasks/{taskId}"

    # Query parameters
    params = {
        "fields[]": [
            "core", "custom_fields", "points_estimate", "time_estimate", "time_tracking",
            "comment_mentions", "comment_attachments", "editor_token", "permissions",
            "statuses", "links", "features", "subtasks_count", "space_id", "folder_id",
            "comment_count", "commit_counts", "direct_parent", "users", "goals",
            "custom_type", "groups", "nested_level"
        ],
        "filterOmit": "task(lower_text_content)"
    }
    
    # Headers
    headers = {
        "Authorization": "Bearer <access token>",
        "sec-ch-ua-platform": "\"Windows\"",
        "X-load-event": "RouterEffects taskIdParamChanged",
        "Referer": "https://app.clickup.com/t/"+taskId,
        "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
        "X-Build-Git-Sha-Short": "c391b09",
        "sec-ch-ua-mobile": "?0",
        "X-CSRF": "1",
        "build-version": "3.80.4",
        "X-Workspace-ID": "1854334",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "sessionId": "gqovwpy0o"
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    # Output response
    if response.status_code == 200:
        print("Success!")
        #print(response.json())  # or response.text if not JSON
    else:
        print(f"Error {response.status_code}: {response.text}")
    return response.json()

def main():
    datas = load_json_file("D:\personal\genai\clickup\\first_api_response.json")
    division_array = datas.get("list").get("divisions")
    taskIds = []
    for division in division_array:
        for group in division.get("groups"):
            if group.get("task_ids"):
                taskIds.extend(group.get("task_ids"))
    

    taskDetailList=[]
    for taskId in taskIds:
        
        response = fetch_task_details_byId(taskId)
        task = response.get("task")
        task_detail = {
            "id":task.get("id"),
            "title":task.get("name"),
            "content":task.get("text_content")
        }
        print(task_detail)
        taskDetailList.append(task_detail)

    # Initialize vector database
    vector_db = VectorDBDemo()

    documents, metadatas, ids = vector_db.prepare_documents(taskDetailList)
            
    # Store in vector database
    vector_db.store_in_vector_db(documents, metadatas, ids)
    
        
    
    
    # Demo search functionality
    print(f"\n{'='*60}")
    print("DEMO: Vector Search")
    print('='*60)
    
    # Example searches
    search_queries = [
        "What detailed should be visible in member pannel?"

    ]
    
    for query in search_queries:
        results = vector_db.search_similar(query, n_results=3)
        callGenAI(results, query)
        print("----------- Done------------------")

if __name__ == "__main__":
    # Required packages (install with pip):
    required_packages = [
        "chromadb",
        "sentence-transformers", 
        "requests"
    ]
    
    print("Required packages:")
    for package in required_packages:
        print(f"  pip install {package}")
    print()
    
    main()
