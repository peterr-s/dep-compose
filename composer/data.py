import os

import conllu
import torch

from composer.utils import find_nps_in_tree

class ComposerCONLLIterableDataset(torch.utils.data.IterableDataset) :
    def __init__(self,
            paths,
            token_id_getter,
            token_presence_checker) :
        self.paths = paths
        self.token_id_getter = token_id_getter
        self.token_presence_checker = token_presence_checker
        self._dep_to_id = dict()
        self._id_to_dep = list()

    def __iter__(self) :
        for path in self.paths :
            with open(path) as input_file :
                for sentence in conllu.parse_incr(input_file) :
                    for token in sentence :
                        dep = token.get("deprel")
                        if dep not in self._dep_to_id :
                            self._dep_to_id[dep] = len(self._id_to_dep)
                            self._id_to_dep.append(dep)

                    for np in find_nps_in_tree(sentence.to_tree()) :
                        token_list = np.to_list()

                        form_list = [token.get("form") for token in token_list]
                        target_token = "_".join(form_list)
                        if not self.token_presence_checker(target_token) :
                            continue

                        input_ids = torch.LongTensor([self.token_id_getter(form)
                                for form in form_list])
                        output_id = torch.LongTensor([self.token_id_getter(target_token)])

                        dep_ids = torch.LongTensor([self._dep_to_id[token.get("deprel")]
                                for token in token_list])

                        id_to_idx = {token.get("id") : i for i, token in enumerate(token_list)}
                        head_idcs = torch.LongTensor([id_to_idx.get(token.get("head"), 0)
                                for token in token_list])

                        yield input_ids, dep_ids, head_idcs, output_id
