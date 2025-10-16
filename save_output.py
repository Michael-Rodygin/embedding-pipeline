def save_embeddings(embeddings, output_file, separator="$$$"):
    """
    Save embeddings to a text file with the specified separator.

    Args:
        embeddings (list): List of embeddings (each can be list, array, or string)
        output_file (str): Path to the output file
        separator (str): Separator string (default: "$$$")

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for i, embedding in enumerate(embeddings):
                # Convert embedding to string representation
                if isinstance(embedding, (list, tuple)):
                    # For list/tuple embeddings, convert to comma-separated string
                    embedding_str = ",".join(str(x) for x in embedding)
                else:
                    # For other types (numpy arrays, etc.), use string representation
                    embedding_str = str(embedding)

                file.write(embedding_str)

                # Add separator except after the last embedding
                if i < len(embeddings) - 1:
                    file.write(separator)

        print(f"Successfully saved {len(embeddings)} embeddings to {output_file}")
        return True

    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False


def save_embeddings_with_metadata(embeddings, output_file, metadata=None, separator="$$$"):
    """
    Save embeddings with optional metadata.

    Args:
        embeddings (list): List of embeddings
        output_file (str): Path to the output file
        metadata (list): Optional list of metadata strings for each embedding
        separator (str): Separator string (default: "$$$")

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            for i, embedding in enumerate(embeddings):
                # Add metadata if provided
                if metadata and i < len(metadata):
                    file.write(f"METADATA:{metadata[i]}|EMBEDDING:")

                # Convert embedding to string representation
                if isinstance(embedding, (list, tuple)):
                    embedding_str = ",".join(str(x) for x in embedding)
                else:
                    embedding_str = str(embedding)

                file.write(embedding_str)

                # Add separator except after the last embedding
                if i < len(embeddings) - 1:
                    file.write(separator)

        print(f"Successfully saved {len(embeddings)} embeddings with metadata to {output_file}")
        return True

    except Exception as e:
        print(f"Error saving embeddings: {e}")
        return False

