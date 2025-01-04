import os


### NLTK ###
import nltk
if not os.path.exists(os.path.join(nltk.data.find('tokenizers'), 'punkt')):
    nltk.download('punkt')

def nltk_sent_tokenize(texts: list[str]):
    return (sent for text in texts for sent in nltk.sent_tokenize(text))


# ### Spacy ###
# import spacy
# try:
#     spacy_nlp = spacy.load('en_core_web_sm')
# except OSError:
#     spacy.cli.download("en_core_web_sm")
#     spacy_nlp = spacy.load('en_core_web_sm')

# def spacy_sent_tokenize(texts: list[str]):
#     # nlp = spacy.load('en_core_web_sm')
#     return (sent.text for text in texts for sent in spacy_nlp(text).sents)


# ### Segtok ###
# from segtok.segmenter import split_single, split_multi

# def segtok_sent_tokenize(texts: list[str]):
#     return (sent for text in texts for sent in split_single(text))


### Sentence Tokenization ###

def sent_tokenize(text, method: str = 'nltk', initial_split_sep: str = None) -> list[str]:
    def has_info(text: str):
        return any(char.isalnum() for char in text)

    texts = [text] if isinstance(text, str) else text
    assert isinstance(texts, list)

    if initial_split_sep:
        texts = [sline 
                 for text in texts 
                 for line in text.split(initial_split_sep) 
                 if (sline := line.strip())]

    if method == 'nltk':
        sents = nltk_sent_tokenize(texts)
    # elif method == 'spacy':
    #     sents = spacy_sent_tokenize(texts)
    # elif method == 'segtok':
    #     sents = segtok_sent_tokenize(texts)
    elif method == 'none':
        sents = texts
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return [ssent for sent in sents if (ssent := sent.strip()) and has_info(ssent)]