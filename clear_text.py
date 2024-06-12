import re

def remove_punctuation(text):
    if text is None:
        return ' '
    return re.sub(r'[^\w\s]', '', text).lower()