import os

import conllu
from gensim.corpora.textcorpus import TextCorpus

CONLL_EXTENSIONS = {".conll", ".conllu", ".conllx"}

class CONLLCorpus(TextCorpus) :
    def __init__(self, input = None) :
        if input is None :
            input = "."
        self.input = input

    def get_texts(self) :
        paths = [os.path.join(directory, filename)
                for directory, _, files in os.walk(self.input)
                for filename in files
                if os.path.splitext(filename)[1] in CONLL_EXTENSIONS]

        for path in paths :
            with open(path) as input_file :
                for sentence in conllu.parse_incr(input_file) :
                    yield [token.get("form") for token in sentence]
