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
            if len(text) > 2000 :
                text = self.getDocumentSummary(text)
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
    
    def getDocumentSummary(self, document: str) -> str:
        """
        Generate a summary for a given document using an external API
        """
        url = "http://localhost:1234/v1/chat/completions"
        prompt = f"""Persona: Act as an expert Technical Writer specializing in medical software documentation. Your task is to create "embedding-optimized summaries."
Primary Goal: You will be given a document describing a feature or user story for a medical software application. Your goal is to generate a dense, factual summary of this document, specifically optimized for semantic search in a vector database.
Core Task Breakdown:
Analyze: Read the provided source document.
Extract Key Entities: Identify and list the core components:
User Persona/Role: (e.g., Oncologist, Radiologist, Clinical Administrator)
Core Action/Functionality: (e.g., creating a treatment plan, annotating a DICOM image, managing patient consent forms)
Clinical Context/Data: (e.g., chemotherapy protocols, MRI scans, HIPAA compliance)
Software Module/System: (e.g., EMR/EHR system, a new reporting dashboard, the scheduling module)
Desired Outcome/Benefit: (e.g., improve diagnostic accuracy, streamline billing, reduce patient wait times)
Synthesize Summary: Weave the extracted entities into a brief, descriptive paragraph. The summary should state what the feature is, who it's for, and what its purpose is, using the specific terminology from the document.
Key Constraints:
Brevity: Strictly limit the summary to 2000 char only (350 words only).
Density: Avoid conversational language or filler words. Every word should contribute to the document's semantic meaning for better searchability.
Factual: Do not infer or add information not present in the source document.
Example:
Source Document: "As a cardiologist, I want to be able to access a patient's historical ECG and Echocardiogram reports directly from the main dashboard, so that I can quickly compare them with the latest readings during a consultation without navigating to a separate archive system."
Ideal Summary: "Feature for cardiologists to access historical ECG and Echocardiogram reports from the main patient dashboard. This allows for quick comparison with current readings during consultations, improving diagnostic efficiency by eliminating the need to search a separate archive."

# output format
provide the summary in the following json format:
<summary>summary text</summary>

# Your Input Document is below:
{document}
"""
        payload = json.dumps({
        "model": "deepseek-r1-distill-llama-8b",
        "messages": [
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
        
        if response.status_code != 200:
            raise Exception(f"Error fetching summary: {response.status_code} - {response.text}")
        response_data = response.json()
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            raise Exception("Invalid response format from summary API")
        summary = response_data['choices'][0]['message']['content'] 
        print(f"SUmmary : {summary}")
        start_index =str.find(summary,"<summary>")
        if start_index> -1 :
            start_index+=9
            end_index = str.find(summary,"</summary>", start_index)
            summary=summary[start_index:end_index]
        
        
        print(f"Response : {response.text}")
        print(f"Generated Summary: {summary}")
        return summary
    
    
class VectorDBDemo:
    def __init__(self):
        # Initialize ChromaDB client (persistent storage)
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db_lm_studio_with_summary")
        
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
        DISTANCE_THRESHOLD = 0.69  # Adjust based on your needs
        
        # Generate embedding for query
        query_embedding = self.embedding_model.encode([query])
        
        # Search in vector database
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        lastIndex = 0
        for distance in results.get("distances")[0]:
            if distance < DISTANCE_THRESHOLD:
                lastIndex += 1
            else:
                break
        if lastIndex == 0:
            print("No results found within the distance threshold")
            lastIndex = 1
            #raise ValueError("No results found within the distance threshold for query : "+query)
            
        results['documents'][0] = results['documents'][0][:lastIndex]
        results['metadatas'][0] = results['metadatas'][0][:lastIndex]
        results['distances'][0] = results['distances'][0][:lastIndex]
        results['ids'][0] = results['ids'][0][:lastIndex]
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

def callGenAI(results: Dict, query) -> str:
        """
        Display search results in a readable format
        """
        if not results['documents'] or not results['documents'][0]:
            print("No results found")
            return
        
        print("\n🔍 Search Results:")
        print("-" * 50)
        
        
        prompt = f"Query: {query}\n\n"
        prompt += "Here are the relevant story documents:\n"
        for i, (doc, metadata, distances, ids) in enumerate(zip(
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
            "content": """You are an AI based business Analyst for health care software development team. You have given software development story document.\n\n
            Your job is to Read and understand business requirement from document and based on taht answer given query, also attached story_id as an reference with your answer.
            
            Strickly follow below output format.
            
            # output format
            <answer> your <reference>StoryId</reference> answer </answer>"""
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
        return response.text
import requests

def fetch_task_ids_by_sprintId(sprintId : str) :
    import requests
    import json

    url = "https://frontdoor-prod-ap-southeast-2-2.clickup.com/view/v1/genericView?fresh_req=false&available_rollups=true"

    payload = json.dumps({
    "id": f"6-{sprintId}-1",
    "members": [],
    "group_members": [],
    "name": "List",
    "parent": {
        "id": sprintId,
        "type": 6
    },
    "type": 1,
    "creator": 49440988,
    "pinned": False,
    "me_view": False,
    "locked": False,
    "visibility": 1,
    "settings": {
        "show_task_locations": False,
        "show_subtasks": 1,
        "show_subtask_parent_names": False,
        "show_closed_subtasks": False,
        "show_assignees": False,
        "show_images": False,
        "show_timer": False,
        "collapse_empty_columns": False,
        "me_comments": True,
        "me_subtasks": True,
        "me_checklists": True,
        "show_empty_statuses": False,
        "auto_wrap": False,
        "time_in_status_view": 1,
        "is_description_pinned": False,
        "override_parent_hierarchy_filter": False,
        "fast_load_mode": False,
        "show_task_properties": True,
        "show_sprint_cards": {
        "show_add_tasks": True,
        "show_add_estimates": True,
        "show_add_assignees": True
        },
        "show_empty_fields": False,
        "field_rendering": 1,
        "colored_columns": True,
        "card_size": 2,
        "task_cover": 2
    },
    "embed_settings": None,
    "grouping": {
        "field": "status",
        "dir": 1,
        "collapsed": [],
        "ignore": False,
        "single": False
    },
    "divide": {
        "field": None,
        "dir": None,
        "collapsed": [],
        "by_subcategory": None
    },
    "sorting": {
        "fields": []
    },
    "frozen_by": {
        "id": 0,
        "username": None,
        "email": None,
        "color": None,
        "initials": None,
        "profilePicture": None
    },
    "filters": {
        "search": "",
        "show_closed": True,
        "search_custom_fields": True,
        "search_description": True,
        "search_name": True,
        "op": "AND",
        "filter_group_ops": [],
        "filter_groups": [],
        "fields": []
    },
    "columns": {
        "fields": [
        {
            "field": "assignee",
            "width": 160,
            "hidden": False,
            "name": None
        },
        {
            "field": "dueDate",
            "width": 160,
            "hidden": False,
            "name": None
        },
        {
            "field": "priority",
            "width": 160,
            "hidden": False,
            "name": None
        },
        {
            "field": "name",
            "width": 1027,
            "hidden": True,
            "name": None
        },
        {
            "field": "status",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "id",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "customId",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "dateCreated",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "dateDone",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "dateUpdated",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "startDate",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "duration",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "timeLoggedRollup",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "timeEstimateRollup",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "sprints",
            "width": 185,
            "hidden": True,
            "name": None
        },
        {
            "field": "pointsEstimate",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "dateClosed",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "createdBy",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "latestComment",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "commentCount",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "lists",
            "width": 185,
            "hidden": True,
            "name": None
        },
        {
            "field": "pullRequests",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "incompleteCommentCount",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "timeInStatus",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "linked",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "dependencies",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "pages",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "cf_6fc69b80-7054-434c-a80b-df5be1ad2e58",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "cf_06d9698f-9b5c-45a1-86bb-c6e936335411",
            "width": 160,
            "hidden": True,
            "name": None
        },
        {
            "field": "cf_da1a53a7-408d-49df-a79c-740d3ebf7287",
            "width": 160,
            "hidden": True,
            "name": None
        }
        ]
    },
    "default": False,
    "standard": True,
    "standard_view": True,
    "orderindex": 1,
    "public": False,
    "seo_optimized": False,
    "public_duplication_enabled": False,
    "tasks_shared_with_me": False,
    "team_sidebar": {
        "assigned_comments": False,
        "assignees": [],
        "group_assignees": [],
        "unassigned_tasks": False
    },
    "frozen_note": None,
    "public_share_expires_on": None,
    "share_tasks": True,
    "share_task_fields": [
        "assignees",
        "priority",
        "due_date",
        "content",
        "comments",
        "attachments",
        "customFields",
        "subtasks",
        "tags",
        "checklists",
        "coverimage"
    ],
    "board_settings": {},
    "team_id": "1854334",
    "sidebar_view": False,
    "sidebar_orderindex": None,
    "sidebar_num_subcats_between": 0,
    "doc_type": 1
    })
    headers = {
    'accept': 'application/json, text/plain, */*',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InNiVkFxWkNGdVJBPSJ9.eyJ1c2VyIjoyMTQ5MzU0LCJ2YWxpZGF0ZWQiOnRydWUsIndzX2tleSI6NTc5NjA4NTA4OSwic2Vzc2lvbl90b2tlbiI6dHJ1ZSwid29ya3NwYWNlcyI6W3sidHlwZSI6InBhc3N3b3JkIn1dLCJpYXQiOjE3NTAzMTExODYsImV4cCI6MTc1MDQ4Mzk4Nn0.vDi4ppjucj4iHBxHA0TgC5EER8dmi5jk9fFVAEJRyyI',
    'build-version': '3.83.0',
    'content-type': 'application/json',
    'origin': 'https://app.clickup.com',
    'priority': 'u=1, i',
    'referer': 'https://app.clickup.com/1854334/v/l/6-901606608101-1?pr=19216249',
    'sec-ch-ua': '"Google Chrome";v="137", "Chromium";v="137", "Not/A)Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'sessionid': 'egutb6fmf',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
    'x-build-git-sha-short': '9a8ab8d',
    'x-csrf': '1',
    'x-fedl-skip-cache': '?1',
    'x-reload-event': 'getDataViewReloadData',
    'x-workspace-id': '1854334',
    'Cookie': '_ga=GA1.1.1146313626.1750311109; ajs_anonymous_id=1a835e1e-b395-47c1-b340-1c60a01245ac; ajs_user_id=2149354; cu-redirect-to-app=true; ajs_group_id=1854334; cu_form_jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InNiVkFxWkNGdVJBPSJ9.eyJ1c2VyIjoyMTQ5MzU0LCJ2YWxpZGF0ZWQiOnRydWUsIndzX2tleSI6NTc5NjA4NTA4OSwiZm9ybSI6dHJ1ZSwic2Vzc2lvbl90b2tlbiI6dHJ1ZSwid29ya3NwYWNlcyI6W3sidHlwZSI6InBhc3N3b3JkIn1dLCJpYXQiOjE3NTAzMTUwNDIsImV4cCI6MTc1MDQ4Nzg0Mn0.rkrp6zNW9pNT7Lqprr9uXtF9LSFPcLdTGN8jnK9iHAY; analytics_session_id=1750315046006; analytics_session_id.last_access=1750316519414; _ga_CMNNJGSJVV=GS2.1.s1750315045$o2$g1$t1750316520$j60$l0$h0'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

    # Output response
    if response.status_code == 200:
        print("Success!")
        #print(response.json())  # or response.text if not JSON
    else:
        print(f"Error {response.status_code}: {response.text}")
    return response.json()



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
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6InNiVkFxWkNGdVJBPSJ9.eyJ1c2VyIjoyMTQ5MzU0LCJ2YWxpZGF0ZWQiOnRydWUsIndzX2tleSI6NTc5NjA4NTA4OSwic2Vzc2lvbl90b2tlbiI6dHJ1ZSwid29ya3NwYWNlcyI6W3sidHlwZSI6InBhc3N3b3JkIn1dLCJpYXQiOjE3NTAzMTExODYsImV4cCI6MTc1MDQ4Mzk4Nn0.vDi4ppjucj4iHBxHA0TgC5EER8dmi5jk9fFVAEJRyyI",
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
    # Initialize vector database
    vector_db = VectorDBDemo()

    """datas = load_json_file("D:\personal\genai\clickup\\first_api_response.json")
    
    901606608101
    901607028210
    901607372737
    """
    datas = fetch_task_ids_by_sprintId("901607706672901607706672")


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
    
    """for query in search_queries:
        results = vector_db.search_similar(query, n_results=3)
        callGenAI(results, query)
        print("----------- Done------------------")"""

def search_task(query : str) -> str:
    """
    Search for tasks based on query
    """
    vector_db = VectorDBDemo()
    
    # Search in vector database
    results = vector_db.search_similar(query, n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found")
        return
    
    # Call GenAI to process results
    response = "Here are the relevant story documents:\n"
    for i, (doc, metadata, distances, ids) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0], 
        results['distances'][0],
        results['ids'][0]
    )):
        response += f"\n\nStory id = {ids} \n story content : {json.dumps(metadata, indent=2)}"
    
        


    print(f"Response from GenAI: {response}")
    return response

def query_task(query : str) -> str:
    """
    Search for tasks based on query
    """
    vector_db = VectorDBDemo()
    
    # Search in vector database
    results = vector_db.search_similar(query, n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        print("No results found")
        return
    
    # Call GenAI to process results
    response = callGenAI(results, query)
    response.json()
    print(f"Response from GenAI: {response}")
    return response


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
