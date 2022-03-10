import re
import spacy
from sacremoses import MosesTokenizer
# spacy_en = spacy.load('en_core_web_sm')
mtok = MosesTokenizer(lang='en')


def tok_tweet_text(tweet, is_tokenize, remove_tag, remove_stop_word=False, concrete_word_only=False):
    global stopwords

    if is_tokenize:
        tweet = tweet.lower()  # lower case

    if remove_tag:
        tweet = re.sub(r'@(\S+)', r'', tweet)  # @handle
        tweet = re.sub(r'#(\S+)', r'', tweet)  # #hashtag
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', r'', tweet)  # URLs
        tweet = preprocess_emojis(tweet, remove_tag)  # Replace emojis with either EMO_POS or EMO_NEG
    else:
        # tweet = re.sub(r'@(\S+)', r'USER_\1', tweet)  # @handle
        tweet = re.sub(r'@(\S+)', r'', tweet)  # @handle
        tweet = re.sub(r'#(\S+)', r' \1 ', tweet)  # #hashtag
        tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)  # URLs
        tweet = preprocess_emojis(tweet, remove_tag)  # Replace emojis with either EMO_POS or EMO_NEG
    tweet = re.sub(r'\brt\b', '', tweet)  # Remove RT
    tweet = re.sub(r'\.{2,}', ' ', tweet)  # Replace 2+ dots with space
    tweet = tweet.strip(' "\'')  # Strip space, " and ' from tweet
    tweet = re.sub(r'\s+', ' ', tweet)  # Replace multiple spaces with a single space

    if not remove_stop_word:
        stopwords = []

    processed_tweet = []
    for word in tweet.split():
        word = preprocess_word(word)
        if is_valid_word(word) and word not in stopwords:
            processed_tweet.append(word)
    tweet = ' '.join(processed_tweet)

    if is_tokenize:
        if concrete_word_only:
            tokens = [tok.lemma_ for tok in mtok.tokenize(tweet)
                      if tok.lemma and (tok.pos_ == 'VERB' or tok.pos_ == 'ADJ' or tok.pos_ == 'NOUN')]
        else:
            tokens = [tok for tok in mtok.tokenize(tweet)]
        return ' '.join(tokens)
    else:
        return tweet


def preprocess_emojis(tweet, remove=False):
    if remove:
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', '', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', '', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', '', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', '', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', '', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', '', tweet)
    else:
        # Smile -- :), : ), :-), (:, ( :, (-:, :')
        tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
        # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
        tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
        # Love -- <3, :*
        tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
        # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
        tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
        # Sad -- :-(, : (, :(, ):, )-:
        tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
        # Cry -- :,(, :'(, :"(
        tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def is_valid_word(word):
    return re.search(r'^[a-zA-Z][a-z0-9A-Z._]*$', word) is not None  # Check if word begins with an alphabet


def preprocess_word(word):
    word = word.strip('“”\'"?!,.():;‘’')  # Remove punctuation
    word = re.sub(r'(.)\1+', r'\1\1', word)  # Convert >= 2 letter repetitions to 2 letter, e.g., funnnnny --> funny
    word = re.sub(r'[-|\']', '', word)  # Remove - & '
    word = re.sub(
        r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't""",
        'not', word)
    return word


stopwords = set(["semst", "\"", "#", "$", "%", "&", "\\", "'", "(", ")",
                 "*", ",", "-", ".", "/", ":", ";", "<", ">", "@",
                 "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?"] + \
                ["a", "about", "above", "after", "again", "am", "an", "and", "any", "are", "as", "at",
                 "be",
                 "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
                 "does",
                 "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
                 "he",
                 "he'd",
                 "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
                 "i",
                 "i'd",
                 "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more",
                 "most",
                 "my", "myself", "nor", "of", "on", "once", "or", "other", "ought", "our", "ours", "ourselves",
                 "out",
                 "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than",
                 "that",
                 "that's",
                 "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
                 "they'll",
                 "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
                 "we",
                 "we'd", "will",
                 "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which",
                 "while", "who",
                 "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your",
                 "yours",
                 "yourself", "yourselves"])