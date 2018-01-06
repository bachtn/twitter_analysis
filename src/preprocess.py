import sys
import regex as re
import pickle
import numpy as np

world_index = pickle.load(open("world_index.p", "rb"))

FLAGS = re.MULTILINE | re.DOTALL

# remove hashtag and add a tag
def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result

# remove caps and add a tag
def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"

# glove tokenization
def tokenize(text):
    # manage smiley
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # tokenize
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    # url
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"/"," / ")

    # users
    text = re_sub(r"@\w+", "<user>")

    # smiley
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")

    # hashtag
    text = re_sub(r"#\S+", hashtag)

    # repeat
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")

    # caps
    text = re_sub(r"([A-Z]){2,}", allcaps)

    # remove caps
    text = text.lower()

    # split tokens
    text = text.rsplit()

    # add padding and getting world index
    out = np.zeros((50, 1))
    for i in range(len(text)):
        try:
            out[i, 0] = world_index[text[i]]
        except:
            continue

    return out
