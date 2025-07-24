from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import os
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import base64
from io import BytesIO
import tempfile
import asyncio
import aiofiles
import PyPDF2
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

app = FastAPI(title="Voice Bot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
bot_name = "ava"
conversation_history = []
bearer_token = None

# RAG Components
chroma_client = None
embedding_model = None
text_splitter = None
pdf_collection = None

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]
    user_input: str

class SummaryRequest(BaseModel):
    email: str
    summary: str

class AuthRequest(BaseModel):
    api_key: str
    project_id: str

class PDFUploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    chunks_processed: int

# Initialize RAG components on startup
@app.on_event("startup")
async def startup_event():
    global bearer_token, chroma_client, embedding_model, text_splitter, pdf_collection
    
    # Initialize authentication
    api_key = os.getenv("WATSONX_API_KEY")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    
    if api_key and project_id:
        bearer_token = get_bearer_token(api_key)
        if bearer_token:
            print("✅ Authentication successful!")
        else:
            print("❌ Authentication failed!")
    else:
        print("❌ Missing WATSONX_API_KEY or WATSONX_PROJECT_ID in environment variables")
    
    # Initialize RAG components
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Get or create collection
        pdf_collection = chroma_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        print("✅ RAG components initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing RAG components: {e}")

def get_bearer_token(api_key: str) -> Optional[str]:
    """Get bearer token for Watsonx API authentication"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"

    try:
        response = requests.post(url, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            print(f"Failed to retrieve access token: {response.text}")
            return None
    except Exception as e:
        print(f"Error getting bearer token: {e}")
        return None

def clean_ai_response(response_text: str) -> str:
    """Clean the AI response by removing template tags and unwanted text"""
    if not response_text:
        return response_text
    
    # Remove common template tags
    unwanted_patterns = [
        "assistant<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "**",
        "assistant<|end_header_id|>\n\n",
        "assistant<|end_header_id|>\n",
    ]
    
    cleaned_response = response_text
    for pattern in unwanted_patterns:
        cleaned_response = cleaned_response.replace(pattern, "")
    
    # Remove leading/trailing whitespace and newlines
    cleaned_response = cleaned_response.strip()
    
    return cleaned_response

def get_watsonx_response(history: List[Message], user_input: str) -> str:
    """Get response from Watsonx API with RAG context"""
    global bearer_token
    
    if not bearer_token:
        return "Error: Not authenticated with Watsonx API"
    
    # Retrieve relevant context from PDF documents
    relevant_context = retrieve_relevant_context(user_input)
    
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer_token}"
    }

    # Construct the conversation history
    conversation = "".join(
        f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>\n" 
        for msg in history
    )
    
    # Add context if available
    context_prompt = ""
    if relevant_context:
        context_prompt = f"\n\nRelevant information from uploaded documents:\n{relevant_context}\n\n"
    
    conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>\n"

    # Create enhanced prompt with context
    enhanced_prompt = conversation + context_prompt + "above the text is user input. Use the provided context if relevant to answer the question. Give answer within 100words only"

    payload = {
        "input": enhanced_prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 8100,
            "min_new_tokens": 0,
            "stop_sequences": [],
            "repetition_penalty": 1
        },
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "project_id": os.getenv("WATSONX_PROJECT_ID")
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            if "results" in response_data and response_data["results"]:
                raw_response = response_data["results"][0]["generated_text"]
                return clean_ai_response(raw_response)
            else:
                return "Error: 'generated_text' not found in the response."
        else:
            return f"Error: Failed to fetch response from Watsonx.ai. Status code: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def get_conversation_summary(conversation_history: List[Message]) -> str:
    """Generate a summary of the conversation using Watsonx"""
    if not conversation_history:
        return "No conversation to summarize."
    
    # Format conversation for summary
    conversation_text = "\n".join([f"{msg.role}: {msg.content}" for msg in conversation_history])
    
    # Create summary prompt
    summary_prompt = f"""Please provide a concise summary of the following conversation:

{conversation_text}

Summary:"""
    
    # Get summary from Watsonx
    try:
        summary = get_watsonx_response([], summary_prompt)
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def send_to_slack(summary: str, webhook_url: str, bot_name: str = "Ava") -> bool:
    """Send conversation summary to Slack for human agent review"""
    try:
        # Format the message
        message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📬 Message from {bot_name} – Conversation Summary",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"Hello team! :wave:\n\n"
                            f"I just wrapped up a conversation with a customer. Here's a summary for your review:"
                        )
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"> {summary.replace(chr(10), chr(10) + '> ')}"
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"_Generated on {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')} by {bot_name}_"
                        }
                    ]
                }
            ]
        }

        # Send to Slack
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        return True

    except Exception as e:
        print(f"Error sending to Slack: {str(e)}")
        return False

def send_summary_email(summary: str, recipient_email: str) -> str:
    """Send conversation summary via email and Slack"""
    try:
        # Get email configuration from environment variables
        sender_email = os.getenv("EMAIL_SENDER")
        email_password = os.getenv("EMAIL_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")

        if not all([sender_email, email_password]):
            return "Email configuration missing. Please set EMAIL_SENDER and EMAIL_PASSWORD in .env file."

        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"{bot_name} – Your Voice Conversation Summary • {datetime.now().strftime('%B %d, %Y at %I:%M %p')}"

        # Add summary to email body
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; color: #333;">
                <h2 style="color: #4B0082;">Hi there! I'm {bot_name} 👋</h2>
                <p>I've put together a quick summary of our recent conversation. Here's what we discussed:</p>
                <div style="background-color: #f0f0f5; padding: 15px; border-left: 5px solid #4B0082; border-radius: 6px; margin: 20px 0;">
                    {summary}
                </div>
                <p>If anything feels off or you'd like me to clarify more, I'm always here to help!</p>
                <p style="margin-top: 30px;">Chat recorded on <strong>{datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}</strong></p>
                <p>With warm regards,</p>
                <p style="font-size: 16px; font-weight: bold;">{bot_name}<br>
                <span style="font-size: 14px; font-weight: normal;">Your Voice Companion</span></p>
            </body>
        </html>
        """

        msg.attach(MIMEText(body, 'html'))

        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, email_password)
            server.send_message(msg)

        # Send to Slack if webhook URL is configured
        if slack_webhook_url:
            slack_success = send_to_slack(summary, slack_webhook_url)
            if slack_success:
                return "Summary sent successfully to email and Slack!"
            else:
                return "Summary sent to email but failed to send to Slack."
        
        return "Summary sent successfully to email!"
    except Exception as e:
        return f"Error sending summary: {str(e)}"

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

def process_pdf_text(text: str, filename: str) -> List[str]:
    """Process PDF text into chunks for vector storage"""
    try:
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Generate embeddings and store in ChromaDB
        embeddings = embedding_model.encode(chunks)
        
        # Prepare documents for storage
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "filename": filename,
                "chunk_index": i,
                "source": "pdf",
                "uploaded_at": datetime.now().isoformat()
            })
            ids.append(f"{filename}_{i}")
        
        # Add to collection
        pdf_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
        
        return chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def retrieve_relevant_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant context from PDF documents based on query"""
    try:
        if not pdf_collection:
            return ""
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query])
        
        # Search for similar documents
        results = pdf_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        if results['documents'] and results['documents'][0]:
            # Combine relevant chunks
            relevant_chunks = results['documents'][0]
            context = "\n\n".join(relevant_chunks)
            return context
        else:
            return ""
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

# API Routes
@app.get("/")
async def root():
    return {"message": "Voice Bot API is running!"}

@app.get("/status")
async def get_status():
    global bearer_token
    return {
        "authenticated": bearer_token is not None,
        "message_count": len(conversation_history),
        "bot_name": bot_name
    }

@app.post("/authenticate")
async def authenticate(request: AuthRequest):
    global bearer_token
    bearer_token = get_bearer_token(request.api_key)
    
    if bearer_token:
        # Set environment variables for this session
        os.environ["WATSONX_API_KEY"] = request.api_key
        os.environ["WATSONX_PROJECT_ID"] = request.project_id
        return {"success": True, "message": "Authentication successful!"}
    else:
        raise HTTPException(status_code=401, detail="Authentication failed")

@app.post("/chat")
async def chat(request: ConversationRequest):
    global conversation_history
    
    # Add user message to history
    conversation_history.append(Message(role="user", content=request.user_input))
    
    # Get AI response
    ai_response = get_watsonx_response(conversation_history[:-1], request.user_input)
    
    if ai_response and not ai_response.startswith("Error"):
        # Add AI response to history
        conversation_history.append(Message(role="assistant", content=ai_response))
        
        return {
            "success": True,
            "response": ai_response,
            "conversation_history": conversation_history
        }
    else:
        raise HTTPException(status_code=500, detail=f"AI Error: {ai_response}")

@app.get("/conversation")
async def get_conversation():
    return {"conversation_history": conversation_history}

@app.post("/clear-conversation")
async def clear_conversation():
    global conversation_history
    conversation_history = []
    return {"success": True, "message": "Conversation cleared!"}

@app.post("/generate-summary")
async def generate_summary():
    summary = get_conversation_summary(conversation_history)
    return {"summary": summary}

@app.post("/send-summary")
async def send_summary(request: SummaryRequest):
    result = send_summary_email(request.summary, request.email)
    return {"result": result}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        text_content = extract_text_from_pdf(file)
        chunks = process_pdf_text(text_content, file.filename)
        return PDFUploadResponse(
            success=True,
            message=f"PDF '{file.filename}' processed successfully. {len(chunks)} chunks generated.",
            filename=file.filename,
            chunks_processed=len(chunks)
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")

@app.post("/retrieve-context")
async def retrieve_context(query: str = "What was the main topic of the conversation?"):
    context = retrieve_relevant_context(query)
    return {"query": query, "context": context}

@app.get("/pdf-documents")
async def get_pdf_documents():
    """Get information about uploaded PDF documents"""
    try:
        if not pdf_collection:
            return {"documents": [], "total_chunks": 0}
        
        # Get all documents from collection
        results = pdf_collection.get()
        
        if not results['documents']:
            return {"documents": [], "total_chunks": 0}
        
        # Group by filename
        documents_info = {}
        for i, doc in enumerate(results['documents']):
            filename = results['metadatas'][i]['filename']
            if filename not in documents_info:
                documents_info[filename] = {
                    "filename": filename,
                    "chunks": 0,
                    "uploaded_at": results['metadatas'][i].get('uploaded_at', 'Unknown')
                }
            documents_info[filename]['chunks'] += 1
        
        return {
            "documents": list(documents_info.values()),
            "total_chunks": len(results['documents'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving PDF documents: {str(e)}")

@app.delete("/pdf-documents/{filename}")
async def delete_pdf_document(filename: str):
    """Delete a PDF document and all its chunks from the knowledge base"""
    try:
        if not pdf_collection:
            raise HTTPException(status_code=404, detail="No PDF collection found")
        
        # Get all documents to find chunks for this filename
        results = pdf_collection.get()
        
        if not results['documents']:
            raise HTTPException(status_code=404, detail="No documents found")
        
        # Find all IDs for chunks of this filename
        ids_to_delete = []
        for i, metadata in enumerate(results['metadatas']):
            if metadata['filename'] == filename:
                ids_to_delete.append(results['ids'][i])
        
        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        
        # Delete the chunks
        pdf_collection.delete(ids=ids_to_delete)
        
        return {
            "success": True,
            "message": f"Document '{filename}' and {len(ids_to_delete)} chunks deleted successfully",
            "deleted_chunks": len(ids_to_delete)
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting PDF document: {str(e)}")



@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 