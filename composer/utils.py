import os
import logging
from typing import List

import conllu

CONLL_EXTENSIONS = {".conll", ".conllu", ".conllx"}
NP_DEPRELS = {"nsubj", "obj", "iobj", "obl", "vocative", "expl", "dislocated", "nmod", "appos", "nummod"}

def configure_logging() :
    logging.basicConfig(
            format = "[%(levelname)s] %(message)s",
            level = logging.DEBUG)

def get_conll_file_paths(root: str) -> List[str] :
    return [os.path.join(directory, filename)
            for directory, _, files in os.walk(root)
            for filename in files
            if os.path.splitext(filename)[1] in CONLL_EXTENSIONS]

def find_nps_in_tree(tree: conllu.models.TokenTree) -> List[conllu.models.TokenTree] :
    nps = [tree] if tree.token.get("deprel") in NP_DEPRELS else list()

    for child in tree.children :
        nps.extend(find_nps_in_tree(child))

    return nps
