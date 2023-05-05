import conllu
from gensim.corpora.textcorpus import TextCorpus

from composer.utils import get_conll_file_paths

class CONLLCorpus(TextCorpus) :
    def __init__(self, input = None) :
        if input is None :
            input = "."
        self.input = input

    def get_texts(self) :
        paths = get_conll_file_paths(self.input)

        for path in paths :
            with open(path) as input_file :
                for sentence in conllu.parse_incr(input_file) :
                    yield [token.get("form") for token in sentence]
