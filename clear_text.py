import re

def remove_punctuation(text):
    if text is None:
        return ' '
    return re.sub(r'[^\w\s]', '', text).lower()


stop_hashteg = [
    '#красивыедевушки',
    '#boobs',
    '#бьюти',
    '#грудь',
    '#as,,'
    '#попа',
    '#girls',
    '#pussy',
    '#bigass'
]

def replace_stop_hashtags(description):
    for hashtag in stop_hashteg:
        description = re.sub(re.escape(hashtag), '', description, flags=re.IGNORECASE)
    return description