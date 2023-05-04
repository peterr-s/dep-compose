import os

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
                sentence = list()
                for line in input_file :
                    line = line.strip()

                    if not line :
                        if sentence :
                            yield sentence
                            sentence = list()
                            continue
                        else :
                            continue
                    if line.startswith("#") :
                        continue

                    fields = line.split("\t")
                    sentence.append(fields[1])

                if sentence :
                    yield sentence
