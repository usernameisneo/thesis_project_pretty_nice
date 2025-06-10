# Project: AI-Powered Thesis Assistant

## Primary Purpose
This project is a comprehensive, AI-powered thesis writing and research management system for master's degree students. It combines a powerful document indexing and search system with an intelligent writing assistant, creating a unified platform to support the entire thesis lifecycle—from research and discovery to writing and citation.

## Core Architecture & Components

The system is built on a modular architecture, ensuring scalability and maintainability.

### 1. Document Processing & Indexing Engine (The Foundation)
- **Multi-format Document Ingestion**: Processes PDFs and other common academic formats using `PyPDF2`, `pdfplumber`, and a multi-pass OCR feature for scanned documents.
- **Intelligent Text Chunking**: Divides documents into semantically meaningful segments for effective analysis and vectorization, using semantic logic, not just keyword and heading matches.
- **Metadata Extraction**: Automatically identifies and extracts titles, authors, abstracts, and bibliographic information.
- **Incremental Processing**: Efficiently processes only new or changed documents to keep the database up-to-date.
- **Advanced Search System**:
    - **Multiple Embedding Models**: Utilizes various sentence transformers (e.g., `SciBERT`, `all-MiniLM-L6-v2`, multilingual models) to capture semantic meaning.
    - **FAISS Vector Database**: Enables high-speed similarity searches across a vast collection of document chunks.
    - **Hybrid Search**: Combines semantic vector search with traditional keyword (TF-IDF) search for comprehensive and accurate results.
    - **Query Expansion & Re-ranking**: Enhances search queries and improves the relevance of search results.
- **Academic API Integration**: Enriches the research database by connecting to:
    - **Semantic Scholar & CrossRef**: For citation counts, author networks, and metadata.
    - **arXiv**: For access to preprint papers.

### 2. AI-Powered Writing & Analysis Assistant (The Intelligence Layer)
- **AI Integration (via OpenRouter)**: Accesses all available AI models by accepting a user-provided API key for diverse analytical and writing tasks.
- **Context-Aware AI Chat**: Allows users to "chat with their documents," asking questions and discussing findings with an AI that understands the research context.
- **AI Writing Assistance**:
    - **Content Generation**: Creates initial drafts and brainstorms ideas, incorporating user feedback.
    - **Academic Tone & Style**: Refines writing to meet academic standards, providing "before and after" suggestions.
    - **Flow & Coherence Analysis**: Improves the logical structure of arguments.
    - **Citation Management**: Assists with formatting citations in APA 7 style.
- **Research Synthesis**: Identifies connections, themes, and research gaps across multiple papers.
- **Citation Recommendation**: Suggests relevant citations based on the context of the writing.

### 3. Thesis & Project Management Module
- **Structured Thesis Projects**: Manages the entire thesis as a project, with chapters, notes, and progress tracking. Includes a feature to preview possible citations and an index of source usage frequency across the thesis.
- **Productivity Tools**: Includes word count tracking and writing session monitoring.
- **Key Data Structures**:
    - `ThesisProject`: Main container with metadata, chapters, and progress.
    - `ThesisChapter`: Individual chapters with content and citations.
    - `WritingSession`: Tracks writing productivity.
    - `CitationEntry`: Manages citations in APA 7 style.
    - `WritingPrompt`: Stores AI prompt templates.

### 4. Multi-Interface Design
- **Professional GUI**: A modern, user-friendly graphical interface built with Tkinter.
    - **Themes**: Includes night mode and dark mode with no white empty areas to reduce eye strain (dark grey/black font or dark background/white font).
    - **Customization**: Allows users to resize font and select colors.
    - **Usability**: Features real-time progress indicators and non-blocking background processing.
- **Command-Line Interface (CLI)**: Offers full functionality for power users and batch processing.
- **Web API**: Provides RESTful endpoints for integration with other research tools.
- **Statistics Dashboard**: Displays analytics about the document collection.
- **Settings Management**: Handles API keys and other user preferences.

## Key Features & Capabilities

### Document Management
- Drag-and-drop PDF processing.
- Automatic duplicate detection.
- File change monitoring for incremental updates.
- Batch processing with progress tracking.
- Export capabilities (CSV, JSON).

### Search & Discovery
- Natural language queries (e.g., "methodology for qualitative research").
- Adjustable similarity thresholds.
- Result clustering and organization.
- Cross-document concept linking.
- Historical search tracking.

## Technical Implementation
- **Language**: Python 3.10+ with comprehensive type hinting.
- **Key Libraries**:
    - **ML/AI**: `sentence-transformers`, `faiss-cpu`, `scikit-learn`, `spacy`
    - **Data**: `pandas`, `numpy`
    - **GUI**: `tkinter`
    - **APIs**: `requests`, `httpx`
- **Code Standards**: Modular design with clear separation of concerns, extensive documentation, and a target of 400-600 lines per file (unless a larger file is necessary).
- **Testing**: A dedicated `tests/` directory with unit tests for all critical functionality.
- **Development Philosophy**: Incremental, Quality-First, User-Centric, AI-Enhanced, and Academic-Focused.

## System Pipelines

- **Search Pipeline**: PDF Input → Text Extraction → Chunking → Embedding Generation → Vector Index → Query Processing → Hybrid Search → Re-ranking → AI Analysis → Results Display.
- **AI Integration Pipeline**: User Query → Document Retrieval → Context Assembly → AI Model Selection → Response Generation → Context Update → Interactive Chat Interface.

## Success Metrics
- **Performance**: Achieve sub-2 second AI response times for interactive features.
- **Lifecycle Support**: Support the complete thesis lifecycle, from planning and research to writing and submission.
- **Scalability**: Capable of managing collections of 10 to 50 scientific academic papers effectively.
- **Usability**: Intuitive and responsive user interface across both GUI and CLI.
