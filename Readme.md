# Simple RAG System Demo

This is a minimalistic Retrieval-Augmented Generation (RAG) system for educational purposes. It demonstrates the key components of a RAG system:

1. Document chunking
2. Embedding generation
3. Semantic retrieval
4. Answer generation

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install openai numpy python-dotenv
```

3. Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Place your documents in the same directory. The code is set up to read from 'paste.txt' and 'paste-2.txt'.

2. Run the script:

```bash
python simple_rag.py
```

3. Enter questions when prompted. The system will:
   - Chunk the documents
   - Create embeddings
   - Retrieve relevant chunks
   - Generate an answer based on those chunks

4. Type 'exit' to quit the demo.

## How It Works

### 1. Document Chunking
Documents are broken into smaller chunks with some overlap to maintain context across chunk boundaries.

### 2. Embedding Generation
Each chunk is converted into a vector embedding using OpenAI's embedding model.

### 3. Retrieval
When a question is asked:
- The question is converted to an embedding
- Similarity is calculated between the question and all chunks
- The most relevant chunks are retrieved

### 4. Answer Generation
The retrieved chunks are provided as context to the language model, which generates an answer based only on this information.

## Educational Notes

This simple RAG system demonstrates how AI can answer questions based on specific documents rather than its pre-trained knowledge. This:
- Improves factual accuracy
- Provides citations for information
- Reduces hallucination
- Enables working with proprietary or recent information

In production systems, you would likely:
- Use a vector database for efficient storage and retrieval
- Implement more sophisticated chunking strategies
- Add additional metadata and filtering
- Employ strategies to handle very large document collections