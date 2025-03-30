from simple_rag import SimpleRAG


def main():
    # Load documents (assuming you've saved the provided content to these files)
    documents = [
        {
            'title': 'Vector Embeddings Guide',
            'content': open('paste.txt', 'r').read()
        },
        {
            'title': 'Retrieval Guide',
            'content': open('paste-2.txt', 'r').read()
        }
    ]

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


if __name__ == "__main__":
    main()