"""Word counting for show_reply and the stopword language heuristic for topics."""

import re

LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
MARKUP = re.compile(r"^\s*(?:```|~~~).*$|[#*>|`~]+|(?<!\w)_+|_+(?!\w)|^\s*(?:[-+]|\d+\.)\s+|^\s*[-:| ]+$", re.MULTILINE)
WORD = re.compile(r"\w", re.UNICODE)

STOPWORDS = {
    "fr": "le la les un une des du de et est pour dans que qui pas sur avec comment je mon ma mes ce cette",
    "en": "the a an and is for in that what how to of with my on this it are do does can",
    "es": "el la los las un una y es para en que qué cómo con mi por del se lo",
    "de": "der die das und ist für in mit wie was ein eine ich mein nicht zu den dem",
    "it": "il lo la gli le un una e è per in che come con mio di del non",
    "pt": "o a os as um uma e é para em que como com meu do da não por",
}
STOPWORD_SETS = {lang: set(words.split()) for lang, words in STOPWORDS.items()}


def count_words(markdown: str) -> int:
    """Whitespace-separated words, ignoring Markdown syntax characters."""
    text = MARKUP.sub(" ", LINK.sub(r"\1", markdown))
    return sum(1 for token in text.split() if WORD.search(token))


def detect_language(topic: str) -> str:
    tokens = re.findall(r"\w+", topic.lower())
    scores = {lang: sum(t in words for t in tokens) for lang, words in STOPWORD_SETS.items()}
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    return best if ranked[0] > 0 and ranked[0] > ranked[1] else "unknown"
