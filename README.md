# Voice Bot with RAG (Retrieval-Augmented Generation)

A sophisticated voice-enabled chatbot that combines real-time speech recognition with RAG (Retrieval-Augmented Generation) capabilities. The bot can process PDF documents and provide contextually relevant answers based on the uploaded content.

## Features

### 🎙️ Voice Interface
- Real-time speech recognition with support for English, Hindi, and Tamil
- Continuous voice chat mode with intelligent pause detection
- Text-to-speech output for AI responses
- Auto-language detection

### 📚 RAG (Retrieval-Augmented Generation)
- PDF document upload and processing
- Vector-based document storage using ChromaDB
- Semantic search and retrieval
- Context-aware responses based on uploaded documents

### 🤖 AI Integration
- Powered by IBM Watsonx LLM (Llama-3-3-70b-instruct)
- Conversation history management
- Intelligent response generation with document context

### 📧 Communication Features
- Conversation summarization
- Email and Slack integration for summaries
- Real-time status monitoring

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **ChromaDB** - Vector database for document storage
- **Sentence Transformers** - Text embedding generation
- **PyPDF2** - PDF text extraction
- **LangChain** - Text processing and chunking

### Frontend
- **React** - User interface framework
- **React Speech Recognition** - Speech-to-text functionality
- **Web Speech API** - Text-to-speech synthesis
- **Axios** - HTTP client for API communication

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- IBM Watsonx API credentials

### Backend Setup
```bash
cd Backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd Frontend
npm install
```

### Environment Configuration
Create a `.env` file in the Backend directory:
```env
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

## Usage

### Starting the Application

1. **Start the Backend:**
```bash
cd Backend
python main.py
```

2. **Start the Frontend:**
```bash
cd Frontend
npm start
```

3. **Access the Application:**
   - Open `http://localhost:3000` in your browser
   - The backend API will be available at `http://localhost:8000`

### Using RAG Features

1. **Upload PDF Documents:**
   - Click "Upload PDF Document" in the PDF Knowledge Base section
   - Select a PDF file from your device
   - The system will automatically process and chunk the document

2. **Ask Questions:**
   - Use voice or text input to ask questions
   - The bot will search through uploaded documents for relevant context
   - Responses will be based on both the conversation history and document content

3. **Monitor Documents:**
   - View uploaded documents in the PDF Knowledge Base section
   - See the number of chunks generated for each document
   - Track total document storage

## API Endpoints

### Core Chat
- `POST /chat` - Send message and get AI response with RAG context
- `GET /conversation` - Get conversation history
- `POST /clear-conversation` - Clear conversation history

### PDF Management
- `POST /upload-pdf` - Upload and process PDF document
- `GET /pdf-documents` - Get list of uploaded documents
- `POST /retrieve-context` - Test context retrieval for a query

### Utilities
- `GET /status` - Check authentication and system status
- `POST /generate-summary` - Generate conversation summary
- `POST /send-summary` - Send summary via email/Slack
- `GET /health` - Health check endpoint

## How RAG Works

1. **Document Processing:**
   - PDF files are uploaded and text is extracted
   - Text is split into chunks using LangChain's RecursiveCharacterTextSplitter
   - Each chunk is converted to embeddings using Sentence Transformers

2. **Vector Storage:**
   - Embeddings are stored in ChromaDB with metadata
   - Documents are indexed for fast semantic search

3. **Query Processing:**
   - User queries are converted to embeddings
   - Similar document chunks are retrieved using cosine similarity
   - Relevant context is combined with the user query

4. **Response Generation:**
   - Context and query are sent to Watsonx LLM
   - AI generates responses based on both conversation history and document context

## Configuration

### Chunking Parameters
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 200 characters
- **Embedding Model:** all-MiniLM-L6-v2

### Retrieval Parameters
- **Top-k Results:** 3 most relevant chunks
- **Similarity Metric:** Cosine similarity

## Troubleshooting

### Common Issues

1. **PDF Upload Fails:**
   - Ensure the file is a valid PDF
   - Check file size limits
   - Verify ChromaDB is properly initialized

2. **Speech Recognition Issues:**
   - Check browser permissions for microphone access
   - Ensure HTTPS is used for production deployments
   - Try different language settings

3. **Authentication Errors:**
   - Verify Watsonx API credentials in .env file
   - Check API key permissions and project access

### Performance Optimization

- For large PDFs, consider increasing chunk size
- Monitor ChromaDB storage usage
- Use SSD storage for better vector search performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 