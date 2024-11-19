from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def chunk_text(text, max_tokens=350, overlap_ratio=0.25):
    if text:
        splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=max_tokens,
            chunk_overlap=int(max_tokens * overlap_ratio),
            length_function=len,
            is_separator_regex=False,
            separators=[".", "\n"],
        )
        # Split the content into chunks
        # chunks = splitter.chunks(content)

        chunks = splitter.split_text(text)

        return chunks
    return []


def is_non_informative(text, min_length=100, max_menu_ratio=0.3):
    # Check text length
    if len(text) < min_length:
        return True

    # Look for common web page elements
    web_elements = ["menu", "search", "home", "you are here"]
    element_count = sum(
        1 for element in web_elements if element.lower() in text.lower()
    )

    # Calculate ratio of web elements to text length
    element_ratio = element_count / len(text.split())

    # Check for excessive newlines, often indicative of menus
    newline_ratio = text.count("\n") / len(text)

    # If many web elements or excessive newlines, likely non-informative
    if element_ratio > max_menu_ratio or newline_ratio > 0.05:
        return True

    # Check for repeated short phrases, often seen in menus
    short_phrases = re.findall(r"\b\w+(?:\s+\w+)?\b", text)
    if len(set(short_phrases)) / len(short_phrases) < 0.7:
        return True

    return False
