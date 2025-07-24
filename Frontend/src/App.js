import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { 
  FaMicrophone, 
  FaMicrophoneSlash, 
  FaTrash, 
  FaEnvelope, 
  FaFileAlt,
  FaPlay,
  FaStop,
  FaRobot,
  FaUser,
  FaVolumeUp,
  FaVolumeMute,
  FaUpload,
  FaFilePdf,
  FaDatabase,
  FaTimes
} from 'react-icons/fa';
import './App.css';

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [summary, setSummary] = useState('');
  const [continuousMode, setContinuousMode] = useState(false);
  const [pdfDocuments, setPdfDocuments] = useState([]);
  const [isUploadingPdf, setIsUploadingPdf] = useState(false);
  const [uploadProgress, setUploadProgress] = useState('');
  
  const { transcript, listening, resetTranscript } = useSpeechRecognition();
  const conversationEndRef = useRef(null);
  const fileInputRef = useRef(null);
  
  // Speech synthesis using browser's built-in API
  const [speechSynthesis, setSpeechSynthesis] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  // Speech detection state
  const [isProcessingSpeech, setIsProcessingSpeech] = useState(false);
  const [speechTimeout, setSpeechTimeout] = useState(null);
  const [lastTranscript, setLastTranscript] = useState('');
  
  // Interruption handling
  const [currentRequest, setCurrentRequest] = useState(null);
  const abortControllerRef = useRef(null);
  
  // Language support
  const [currentLanguage, setCurrentLanguage] = useState('en-US');
  const [availableVoices, setAvailableVoices] = useState([]);

  // Auto-scroll to bottom of conversation
  useEffect(() => {
    conversationEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversationHistory]);

  // Initialize speech synthesis and load voices
  useEffect(() => {
    if ('speechSynthesis' in window) {
      setSpeechSynthesis(window.speechSynthesis);
      
      // Load available voices
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        setAvailableVoices(voices);
      };
      
      // Load voices immediately if available
      loadVoices();
      
      // Listen for voices to be loaded
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  }, []);

  // Check authentication status on component mount
  useEffect(() => {
    checkAuthStatus();
    loadPdfDocuments();
  }, []);

  // Handle speech recognition with intelligent pause detection and interruption
  useEffect(() => {
    if (transcript && isListening) {
      const cleanTranscript = transcript.trim();
      
      // Only process if transcript is not empty and has changed
      if (cleanTranscript && cleanTranscript !== lastTranscript) {
        setLastTranscript(cleanTranscript);
        
        // If bot is currently speaking, interrupt it immediately
        if (isSpeaking) {
          stopSpeaking();
        }
        
        // If there's an ongoing request, cancel it
        if (abortControllerRef.current) {
          abortControllerRef.current.abort();
        }
        
        // Clear any existing timeout
        if (speechTimeout) {
          clearTimeout(speechTimeout);
        }
        
        // Set a timeout to wait for more speech (2 seconds)
        const timeout = setTimeout(() => {
          if (cleanTranscript.length >= 3) {
            setIsProcessingSpeech(true);
            handleVoiceInput(cleanTranscript);
            resetTranscript();
            setLastTranscript('');
            setIsProcessingSpeech(false);
          }
        }, 2000); // Wait 2 seconds after last speech input
        
        setSpeechTimeout(timeout);
      }
    }
  }, [transcript, isListening, isProcessingSpeech, lastTranscript, speechTimeout, isSpeaking]);

  const checkAuthStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      setIsAuthenticated(response.data.authenticated);
      if (response.data.conversation_history) {
        setConversationHistory(response.data.conversation_history);
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
      setIsAuthenticated(false);
    }
  };

  const validateSpeechInput = (text) => {
    const cleanText = text.trim();
    
    // Basic validation - only check length, accept any language
    if (!cleanText || cleanText.length < 3) {
      return { isValid: false, error: 'Speech too short. Please try speaking again.' };
    }
    
    return { isValid: true, text: cleanText };
  };

  const handleVoiceInput = async (text) => {
    // Validate the speech input
    const validation = validateSpeechInput(text);
    
    if (!validation.isValid) {
      setError(validation.error);
      return;
    }
    
    const cleanText = validation.text;
    
    // Check if this is the same as the last message to prevent duplicates
    if (conversationHistory.length > 0 && 
        conversationHistory[conversationHistory.length - 1].content === cleanText) {
      return;
    }

    setIsLoading(true);
    setError('');

    // Create a new AbortController for this request
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    setCurrentRequest(abortController);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        messages: conversationHistory,
        user_input: text
      }, {
        signal: abortController.signal
      });

      if (response.data.success) {
        setConversationHistory(response.data.conversation_history);
        
        // Speak the AI response
        speakText(response.data.response);
        
        setSuccess('Message sent successfully!');
        setTimeout(() => setSuccess(''), 3000);
      }
    } catch (error) {
      // Don't show error if request was aborted (interrupted)
      if (error.name !== 'CanceledError' && error.code !== 'ERR_CANCELED') {
        setError(error.response?.data?.detail || 'Error processing voice input');
      }
    } finally {
      setIsLoading(false);
      setCurrentRequest(null);
      abortControllerRef.current = null;
    }
  };

  const startListening = () => {
    setIsListening(true);
    SpeechRecognition.startListening({ continuous: true, language: 'en-US' });
  };

  const stopListening = () => {
    setIsListening(false);
    setContinuousMode(false);
    SpeechRecognition.stopListening();
    
    // Process any remaining transcript
    if (lastTranscript && lastTranscript.trim().length >= 3) {
      handleVoiceInput(lastTranscript.trim());
    }
    
    // Clear timeouts and reset
    if (speechTimeout) {
      clearTimeout(speechTimeout);
    }
    resetTranscript();
    setLastTranscript('');
    setIsProcessingSpeech(false);
    
    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const startContinuousMode = () => {
    // Reset any previous transcript
    resetTranscript();
    setContinuousMode(true);
    setIsListening(true);
    
    // Use auto-detect if selected, otherwise use specific language
    const language = currentLanguage === 'auto' ? undefined : currentLanguage;
    SpeechRecognition.startListening({ continuous: true, language });
  };

  const clearConversation = async () => {
    try {
      // Stop listening and reset transcript
      if (isListening) {
        SpeechRecognition.stopListening();
        setIsListening(false);
        setContinuousMode(false);
      }
      resetTranscript();
      
      // Stop speaking and cancel any ongoing requests
      stopSpeaking();
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      
      await axios.post(`${API_BASE_URL}/clear-conversation`);
      setConversationHistory([]);
      setSummary('');
      setSuccess('Conversation cleared!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      setError('Error clearing conversation');
    }
  };

  const generateSummary = async () => {
    if (conversationHistory.length === 0) {
      setError('No conversation to summarize');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/generate-summary`);
      setSummary(response.data.summary);
      setSuccess('Summary generated successfully!');
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      setError('Error generating summary');
    } finally {
      setIsLoading(false);
    }
  };

  const sendSummaryEmail = async () => {
    if (!summary) {
      setError('Please generate a summary first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/send-summary`, {
        email: 'ananthananth881@gmail.com',
        summary: summary
      });
      setSuccess(response.data.result);
      setTimeout(() => setSuccess(''), 5000);
    } catch (error) {
      setError('Error sending summary');
    } finally {
      setIsLoading(false);
    }
  };

  const detectLanguage = (text) => {
    // Simple language detection for English, Hindi, and Tamil
    if (/[\u0900-\u097F]/.test(text)) return 'hi-IN'; // Hindi
    if (/[\u0B80-\u0BFF]/.test(text)) return 'ta-IN'; // Tamil
    return 'en-US'; // Default to English
  };

  const speakText = (text) => {
    if (speechSynthesis) {
      const utterance = new SpeechSynthesisUtterance(text);
      
      // Auto-detect language from text if auto is selected, otherwise use current language
      const detectedLanguage = currentLanguage === 'auto' ? detectLanguage(text) : currentLanguage;
      utterance.lang = detectedLanguage;
      
      // Try to find a voice for the detected language
      const voices = availableVoices.filter(voice => 
        voice.lang.startsWith(detectedLanguage.split('-')[0])
      );
      
      if (voices.length > 0) {
        // Use the first available voice for the language
        utterance.voice = voices[0];
      }
      
      utterance.onstart = () => setIsSpeaking(true);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      speechSynthesis.speak(utterance);
    }
  };

  const speakMessage = (text) => {
    speakText(text);
  };

  const submitCurrentSpeech = () => {
    if (lastTranscript && lastTranscript.trim().length >= 3 && !isProcessingSpeech) {
      setIsProcessingSpeech(true);
      handleVoiceInput(lastTranscript.trim());
      resetTranscript();
      setLastTranscript('');
      setIsProcessingSpeech(false);
      
      // Clear timeout
      if (speechTimeout) {
        clearTimeout(speechTimeout);
      }
    }
  };

  const stopSpeaking = () => {
    if (speechSynthesis) {
      speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  const loadPdfDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/pdf-documents`);
      setPdfDocuments(response.data.documents || []);
    } catch (error) {
      console.error('Error loading PDF documents:', error);
    }
  };

  const handlePdfUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Please select a PDF file');
      return;
    }

    setIsUploadingPdf(true);
    setUploadProgress('Uploading PDF...');
    setError('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}/upload-pdf`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(`Processing PDF... ${percentCompleted}%`);
        },
      });

      if (response.data.success) {
        setSuccess(response.data.message);
        setUploadProgress('');
        await loadPdfDocuments(); // Reload the documents list
        setTimeout(() => setSuccess(''), 5000);
      }
    } catch (error) {
      setError(error.response?.data?.detail || 'Error uploading PDF');
      setUploadProgress('');
    } finally {
      setIsUploadingPdf(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const triggerPdfUpload = () => {
    fileInputRef.current?.click();
  };

  const deletePdfDocument = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}"? This action cannot be undone.`)) {
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await axios.delete(`${API_BASE_URL}/pdf-documents/${encodeURIComponent(filename)}`);
      
      if (response.data.success) {
        setSuccess(response.data.message);
        await loadPdfDocuments(); // Reload the documents list
        setTimeout(() => setSuccess(''), 5000);
      }
    } catch (error) {
      setError(error.response?.data?.detail || 'Error deleting PDF document');
    } finally {
      setIsLoading(false);
    }
  };



  return (
    <div className="app-container">
      <div className="main-container">
        <header className="header">
          <h1 className="title">🎙️ Voice Bot with Watsonx LLM</h1>
          <p className="subtitle">Voice Assistant</p>
        </header>

        <div className="content">
          <div className="status-bar">
            <div className={`status-indicator ${isAuthenticated ? 'authenticated' : 'not-authenticated'}`}>
              {isAuthenticated ? '✅ Ready' : '❌ Not Authenticated'}
            </div>
            <div className="status-info">
              <div>💬 Messages: {conversationHistory.length}</div>
              <div className="language-selector">
                <label htmlFor="language-select">🌐 Speech Recognition: </label>
                <select 
                  id="language-select"
                  value={currentLanguage}
                  onChange={(e) => setCurrentLanguage(e.target.value)}
                  disabled={isListening}
                >
                  <option value="auto">Auto Detect</option>
                  <option value="en-US">English (US)</option>
                  <option value="hi-IN">Hindi (India)</option>
                  <option value="ta-IN">Tamil (India)</option>
                </select>
              </div>
            </div>
          </div>

          {error && <div className="error-message">{error}</div>}
          {success && <div className="success-message">{success}</div>}

          {/* PDF Upload Section */}
          <div className="pdf-section">
            <h3>📚 PDF Knowledge Base</h3>
            <div className="pdf-controls">
              <button
                className="btn btn-primary"
                onClick={triggerPdfUpload}
                disabled={isUploadingPdf}
              >
                <FaUpload />
                Upload PDF Document
              </button>
              
              <div className="pdf-info">
                <FaDatabase />
                <span>Documents: {pdfDocuments.length} | Total Chunks: {pdfDocuments.reduce((sum, doc) => sum + doc.chunks, 0)}</span>
              </div>
            </div>
            
            {uploadProgress && (
              <div className="upload-progress">
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
                <span>{uploadProgress}</span>
              </div>
            )}
            
            {pdfDocuments.length > 0 && (
              <div className="pdf-documents-list">
                <h4>Uploaded Documents:</h4>
                {pdfDocuments.map((doc, index) => (
                  <div key={index} className="pdf-document-item">
                    <FaFilePdf />
                    <span className="doc-name">{doc.filename}</span>
                    <span className="doc-chunks">{doc.chunks} chunks</span>
                    <button
                      className="delete-pdf-btn"
                      onClick={() => deletePdfDocument(doc.filename)}
                      title={`Delete ${doc.filename}`}
                      disabled={isLoading}
                    >
                      <FaTimes />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Hidden file input */}
          <input
            type="file"
            ref={fileInputRef}
            onChange={handlePdfUpload}
            accept=".pdf"
            style={{ display: 'none' }}
          />

          <div className="voice-controls">
            <button
              className="btn btn-primary"
              onClick={startContinuousMode}
              disabled={!isAuthenticated || isListening}
            >
              {isLoading ? <div className="loading-spinner" /> : <FaMicrophone />}
              Start Continuous Voice Chat
            </button>

            <button
              className="btn btn-secondary"
              onClick={stopListening}
              disabled={!isListening}
            >
              <FaStop />
              Stop Voice Chat
            </button>

            <button 
              className="btn btn-success" 
              onClick={submitCurrentSpeech}
              disabled={!isListening || !lastTranscript || isProcessingSpeech}
            >
              <FaPlay />
              Send Current Speech
            </button>

            <button className="btn btn-danger" onClick={clearConversation}>
              <FaTrash />
              Clear Conversation
            </button>
          </div>

          {/* Speech Status Indicator */}
          {isListening && (
            <div className="speech-status">
              {lastTranscript ? (
                <div className="speech-detected">
                  <div className="speech-indicator">
                    <FaMicrophone style={{ color: '#28a745', animation: 'pulse 1s infinite' }} />
                    {isSpeaking ? 'Interrupting bot...' : 'Detecting speech...'} (Auto-send in 2s)
                    {currentLanguage === 'auto' && (
                      <span className="auto-detect-badge">🌐 Auto</span>
                    )}
                  </div>
                  <div className="current-transcript">
                    "{lastTranscript}"
                  </div>
                </div>
              ) : (
                <div className="speech-waiting">
                  <FaMicrophone style={{ color: '#6c757d' }} />
                  Waiting for speech...
                  {currentLanguage === 'auto' && (
                    <span className="auto-detect-badge">🌐 Auto</span>
                  )}
                </div>
              )}
            </div>
          )}



          <div className="conversation-container">
            {conversationHistory.length === 0 ? (
              <div className="empty-state">
                <FaRobot size={48} className="empty-state-icon" />
                <p>No conversation yet. Start by clicking 'Start Voice Chat' or speaking!</p>
              </div>
            ) : (
              conversationHistory.map((message, index) => (
                <div className="message-container" key={index}>
                  {message.role === 'user' ? (
                    <>
                      <div style={{ flex: 1 }}></div>
                      <div className="message-bubble user-message">
                        <div>{message.content}</div>
                      </div>
                      <div className="avatar user-avatar">
                        <FaUser />
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="avatar assistant-avatar">
                        <FaRobot />
                      </div>
                      <div className="message-bubble assistant-message">
                        <div>{message.content}</div>
                        <div className="message-actions">
                          {isSpeaking ? (
                            <button className="action-button" onClick={stopSpeaking}>
                              <FaVolumeMute /> Stop
                            </button>
                          ) : (
                            <button className="action-button" onClick={() => speakMessage(message.content)}>
                              <FaVolumeUp /> Speak
                            </button>
                          )}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              ))
            )}
            
            {/* Interruption indicator */}
            {isListening && lastTranscript && isSpeaking && (
              <div className="message-container">
                <div className="avatar assistant-avatar">
                  <FaRobot />
                </div>
                <div className="message-bubble assistant-message interruption-indicator">
                  <div>🔄 Interrupting current response...</div>
                </div>
              </div>
            )}
            
            <div ref={conversationEndRef} />
          </div>

          <div className="summary-section">
            <h3>📊 Conversation Summary</h3>
            <div className="summary-controls">
              <button className="btn btn-success" onClick={generateSummary} disabled={isLoading}>
                <FaFileAlt />
                Generate Summary
              </button>
              <button className="btn btn-primary" onClick={sendSummaryEmail} disabled={!summary || isLoading}>
                <FaEnvelope />
                Send Summary via Email
              </button>
            </div>
            
            {summary && (
              <div className="summary-text">
                <strong>Summary:</strong><br />
                {summary}
              </div>
            )}
          </div>


        </div>
      </div>
    </div>
  );
}

export default App; 