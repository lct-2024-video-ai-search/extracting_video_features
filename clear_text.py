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
    '#ass,'
    '#попа',
    '#girls',
    '#pussy',
    '#bigass',
    '#bigbooty',
    '#ass',
    '#попа',
    '#hotgirl',
    '#купальник',
    '#купальник',
    '#sexygirls',
    
]

def replace_stop_hashtags(description):
    for w in stop_hashteg:
        description = description.replace(w, '')
    return description