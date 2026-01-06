"""
Text processing utilities for json_analyzers.

Provides:
- Word/sentence tokenization
- N-gram extraction
- Discourse marker detection
- Hedge phrase detection
- Text normalization
"""

import re
from typing import List, Set, Tuple

DISCOURSE_MARKERS: Set[str] = {
    "however", "therefore", "moreover", "furthermore", "nevertheless",
    "consequently", "accordingly", "thus", "hence", "meanwhile",
    "otherwise", "instead", "alternatively", "similarly", "likewise",
    "additionally", "also", "besides", "finally", "firstly",
    "secondly", "thirdly", "lastly", "next", "then",
    "and", "but", "or", "yet", "so", "for", "nor",
    "although", "because", "since", "while", "whereas",
    "if", "unless", "until", "when", "where",
    "in conclusion", "in summary", "in contrast", "on the other hand",
    "for example", "for instance", "in addition", "as a result",
}

HEDGE_PHRASES: Set[str] = {
    "maybe", "perhaps", "possibly", "probably", "likely",
    "somewhat", "rather", "quite", "fairly", "relatively",
    "apparently", "seemingly", "supposedly", "allegedly",
    "might", "may", "could", "would", "should",
    "i think", "i believe", "i suppose", "i guess", "i assume",
    "it seems", "it appears", "it looks like", "it might be",
    "in my opinion", "from my perspective", "as far as i know",
    "to some extent", "in a way", "sort of", "kind of",
    "more or less", "up to a point", "in general", "generally speaking",
    "tends to", "seem to", "appear to",
}

ABBREVIATIONS: Set[str] = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "vs", "etc", "ie", "eg", "al", "inc", "ltd", "co",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "st", "nd", "rd", "th",
    "a.m", "p.m", "am", "pm",
}


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into words.
    
    Uses regex to extract alphabetic words, lowercased.
    
    Args:
        text: Input text
        
    Returns:
        List of lowercase words
    """
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences with abbreviation handling.
    
    Handles common abbreviations to avoid false splits.
    
    Args:
        text: Input text
        
    Returns:
        List of sentence strings
    """
    text = normalize_whitespace(text)
    
    abbrev_pattern = r'\b(' + '|'.join(re.escape(a) for a in ABBREVIATIONS) + r')\.'
    placeholder = '\x00ABBREV\x00'
    
    protected = re.sub(abbrev_pattern, r'\1' + placeholder, text, flags=re.IGNORECASE)
    
    protected = re.sub(r'(\d)\.(\d)', r'\1\x01\2', protected)
    
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    parts = sentence_endings.split(protected)
    
    sentences = []
    for part in parts:
        part = part.replace(placeholder, '.')
        part = part.replace('\x01', '.')
        part = part.strip()
        if part:
            sentences.append(part)
    
    if not sentences and text.strip():
        sentences = [text.strip()]
    
    return sentences


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from token list.
    
    Args:
        tokens: List of tokens (words)
        n: N-gram size
        
    Returns:
        List of n-gram tuples
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def get_char_ngrams(text: str, n: int) -> List[str]:
    """Extract character n-grams from text.
    
    Args:
        text: Input text
        n: N-gram size
        
    Returns:
        List of character n-gram strings
    """
    text = text.lower()
    if len(text) < n:
        return []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def extract_discourse_markers(text: str) -> List[str]:
    """Extract discourse markers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of found discourse markers (in order of appearance)
    """
    text_lower = text.lower()
    found = []
    
    multi_word = [m for m in DISCOURSE_MARKERS if ' ' in m]
    for marker in sorted(multi_word, key=len, reverse=True):
        if marker in text_lower:
            found.append(marker)
    
    words = set(tokenize_words(text))
    single_word = [m for m in DISCOURSE_MARKERS if ' ' not in m]
    for marker in single_word:
        if marker in words:
            found.append(marker)
    
    return found


def extract_sentence_templates(sentences: List[str]) -> List[str]:
    """Extract sentence templates by abstracting content words.
    
    Simple heuristic: replace content words with placeholders.
    For full POS-based templates, use spaCy (optional dependency).
    
    Args:
        sentences: List of sentence strings
        
    Returns:
        List of template patterns
    """
    templates = []
    
    for sentence in sentences:
        template = re.sub(r'\b[A-Z][a-z]+\b', 'NOUN', sentence)
        template = re.sub(r'\b\d+\b', 'NUM', template)
        template = re.sub(r'\b[a-z]{7,}\b', 'WORD', template)
        
        templates.append(template)
    
    return templates


def is_hedge_phrase(text: str) -> bool:
    """Check if text contains hedge phrases.
    
    Args:
        text: Input text
        
    Returns:
        True if any hedge phrase is found
    """
    text_lower = text.lower()
    
    for hedge in HEDGE_PHRASES:
        if hedge in text_lower:
            return True
    
    return False


def count_hedge_phrases(text: str) -> int:
    """Count hedge phrases in text.
    
    Args:
        text: Input text
        
    Returns:
        Number of hedge phrases found
    """
    text_lower = text.lower()
    count = 0
    
    for hedge in HEDGE_PHRASES:
        count += len(re.findall(r'\b' + re.escape(hedge) + r'\b', text_lower))
    
    return count


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Collapses multiple spaces/newlines to single space.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_code_blocks(text: str) -> Tuple[str, List[str]]:
    """Extract and remove code blocks from text.
    
    Handles markdown fenced code blocks (```...```) and indented blocks.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (text_without_code, list_of_code_blocks)
    """
    code_blocks = []
    
    fenced_pattern = r'```[\w]*\n(.*?)```'
    fenced_matches = re.findall(fenced_pattern, text, re.DOTALL)
    code_blocks.extend(fenced_matches)
    text_clean = re.sub(fenced_pattern, '', text, flags=re.DOTALL)
    
    lines = text_clean.split('\n')
    clean_lines = []
    current_code_block = []
    in_code_block = False
    
    for line in lines:
        if line.startswith('    ') or line.startswith('\t'):
            in_code_block = True
            current_code_block.append(line)
        else:
            if in_code_block and current_code_block:
                code_blocks.append('\n'.join(current_code_block))
                current_code_block = []
                in_code_block = False
            clean_lines.append(line)
    
    if current_code_block:
        code_blocks.append('\n'.join(current_code_block))
    
    return '\n'.join(clean_lines), code_blocks


def word_overlap(text1: str, text2: str) -> float:
    """Calculate word overlap (Jaccard similarity) between two texts.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Jaccard similarity [0, 1]
    """
    words1 = set(tokenize_words(text1))
    words2 = set(tokenize_words(text2))
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0
