def parse_input(file_path, separator="$$$"):
    """
    Parse input text file using the specified separator.

    Args:
        file_path (str): Path to the input file
        separator (str): Separator string (default: "$$$")

    Returns:
        list: List of parsed text chunks
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Split content using the separator and remove empty strings
        chunks = [chunk.strip() for chunk in content.split(separator) if chunk.strip()]

        print(f"Parsed {len(chunks)} text chunks from {file_path}")
        return chunks

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def parse_input_string(input_string, separator="$$$"):
    """
    Parse input string using the specified separator.

    Args:
        input_string (str): Input string to parse
        separator (str): Separator string (default: "$$$")

    Returns:
        list: List of parsed text chunks
    """
    chunks = [chunk.strip() for chunk in input_string.split(separator) if chunk.strip()]
    print(f"Parsed {len(chunks)} text chunks from input string")
    return chunks


# Example usage
if __name__ == "__main__":
    # Test with a file
    chunks = parse_input("input.txt")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk[:50]}...")
