#!/usr/bin/env python3

import sys
from typing import List

import conllu

NP_DEPRELS = {"nsubj", "obj", "iobj", "obl", "vocative", "expl", "dislocated", "nmod", "appos", "nummod"}

def find_nps_in_tree(tree: conllu.models.TokenTree) -> List[conllu.models.TokenTree] :
    nps = [tree] if tree.token.get("deprel") in NP_DEPRELS else list()

    for child in tree.children :
        nps.extend(find_nps_in_tree(child))

    return nps

def main() :
    for tree in conllu.parse_tree_incr(sys.stdin) :
        print(tree.serialize())

        nps = find_nps_in_tree(tree)
        for np in nps :
            np_form_swap = np.token.get("form")
            np_children_swap = np.children

            np.token["form"] = "_".join([t.get("form") for t in np.to_list()])
            np.children = list()
            print(tree.serialize())

            np.token["form"] = np_form_swap
            np.children = np_children_swap

if __name__ == "__main__" :
    main()
