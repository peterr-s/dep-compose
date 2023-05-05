#!/usr/bin/env python3

import sys

import conllu

from composer.utils import find_nps_in_tree

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
