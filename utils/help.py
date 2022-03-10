import re


def query_text_multi(text, queries):
    if any(q in text for q in queries):
        return True
    else:
        return False
