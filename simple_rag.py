import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SimpleRAG:
    def __init__(self, documents):
        self.documents = documents
        self.embeddings = []
        self.chunks = []

    def chunk_documents(self, chunk_size=200, overlap=50):
        """Simple chunking of documents into smaller pieces"""
        print("1. Chunking documents...")
        self.chunks = []

        for doc in self.documents:
            content = doc['content']
            title = doc['title']

            # Simple chunking by splitting the content
            words = content.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i + chunk_size]
                if len(chunk_words) < 100:  # Skip very small chunks
                    continue

                chunk_text = ' '.join(chunk_words)
                self.chunks.append({
                    'text': chunk_text,
                    'title': title,
                    'doc_index': self.documents.index(doc)
                })

        print(f"Created {len(self.chunks)} chunks from {len(self.documents)} documents")
        return self.chunks

    def embed_chunks(self):
        print("2. Creating embeddings for chunks...")
        self.embeddings = []

        for i, chunk in enumerate(self.chunks):
            print(f"Embedding chunk {i + 1}/{len(self.chunks)}...")

            response = client.embeddings.create(
                input=chunk['text'],
                model="text-embedding-3-small"
            )
            self.embeddings.append(response.data[0].embedding)

        print(f"Created {len(self.embeddings)} embeddings")
        return self.embeddings

    def similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def retrieve(self, query, top_k=3):
        """Retrieve the most relevant chunks for a query"""
        print(f"3. Retrieving relevant chunks for: '{query}'")

        # Create embedding for the query
        query_response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = query_response.data[0].embedding

        # Calculate similarity scores
        similarities = [self.similarity(query_embedding, doc_embedding)
                        for doc_embedding in self.embeddings]

        # Get indices of top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return the top_k chunks with their similarity scores
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'similarity': similarities[idx]
            })

        print(f"Retrieved {len(results)} chunks")
        return results

    def generate_answer(self, query, retrieved_chunks):
        """Generate a response based on the retrieved chunks"""
        print("4. Generating answer based on retrieved chunks...")

        # Format context from retrieved chunks
        context = ""
        for i, result in enumerate(retrieved_chunks):
            context += f"\nCHUNK {i + 1} (Similarity: {result['similarity']:.4f}):\n"
            context += f"From: {result['chunk']['title']}\n"
            context += f"{result['chunk']['text']}\n"

        # Create prompt for GPT
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant that answers questions based only on the provided context. "
                        "If you don't know the answer based on the context, say so."},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer the question based only on the provided context."}
        ]

        # Generate response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )

        return response.choices[0].message.content


# Main function - loads documents and runs the RAG system
def main():
    print("""
    =================================================
    Simple RAG (Retrieval-Augmented Generation) Demo
    =================================================

    This demo shows how RAG works in 4 steps:
    1. Chunking documents into smaller pieces
    2. Creating vector embeddings for each chunk
    3. Retrieving relevant chunks for a query
    4. Generating an answer based on retrieved chunks

    You need an OpenAI API key in a .env file:
    OPENAI_API_KEY=your-api-key-here
    =================================================
    """)

    # Check if file exists
    document_file = 'custom_data.txt'
    if not os.path.exists(document_file):
        print(f"Error: Required file '{document_file}' not found.")
        print("Please create this file with your custom factual data.")
        return

    # Load document
    try:
        document_content = open(document_file, 'r', encoding='utf-8').read()
        documents = [
            {
                'title': 'Custom Knowledge Base',
                'content': document_content
            }
        ]
    except Exception as e:
        print(f"Error loading document: {e}")
        return

    # Create RAG system
    rag = SimpleRAG(documents)

    # Process documents
    rag.chunk_documents()
    rag.embed_chunks()

    # Example query loop
    print("\n=== Simple RAG Demo ===")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'exit':
            break

        # Retrieve relevant chunks
        retrieved_chunks = rag.retrieve(query)

        # Generate answer
        answer = rag.generate_answer(query, retrieved_chunks)

        # Display answer
        print("\n=== Answer ===")
        print(answer)
        print("\n=== End of Answer ===")


if __name__ == "__main__":
    main()