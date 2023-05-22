import os
from typing import Tuple, Generator, Optional, List

import conllu
import torch

from composer.utils import find_nps_in_tree

class CONLLIterableDatasetBase(torch.utils.data.IterableDataset) :
    def __init__(self,
            paths,
            token_id_getter,
            token_presence_checker,
            pad_fn) :
        self.paths = paths
        self.token_id_getter = token_id_getter
        self.token_presence_checker = token_presence_checker
        self.pad_fn = pad_fn
        self._dep_to_id = dict()
        self._id_to_dep = list()

    def __iter__(self) -> Generator[Tuple, None, None] :
        for path in self.paths :
            with open(path) as input_file :
                for sentence in conllu.parse_incr(input_file) :
                    for token in sentence :
                        dep = token.get("deprel")
                        if dep not in self._dep_to_id :
                            self._dep_to_id[dep] = len(self._id_to_dep)
                            self._id_to_dep.append(dep)

                    for np in find_nps_in_tree(sentence.to_tree()) :
                        if (sample := self.create_sample(np)) is not None :
                            yield sample
                        else :
                            continue

    def create_sample(self, *args, **kwargs) :
        raise NotImplementedError()

class ComposerCONLLIterableDataset(CONLLIterableDatasetBase) :
    def create_sample(self, np) -> Tuple[torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor,
            torch.LongTensor] :
        token_list = np.to_list()

        form_list = [token.get("form") for token in token_list]
        target_token = "_".join(form_list)
        if not self.token_presence_checker(target_token) :
            return None

        input_ids = torch.LongTensor([self.token_id_getter(form)
                for form in form_list])
        output_id = torch.LongTensor([self.token_id_getter(target_token)])

        dep_ids = torch.LongTensor([self._dep_to_id[token.get("deprel")]
                for token in token_list])

        id_to_idx = {token.get("id") : i for i, token in enumerate(token_list)}
        head_idcs = torch.LongTensor([id_to_idx.get(token.get("head"), 0)
                for token in token_list])

        return *self.pad_fn(input_ids, dep_ids, head_idcs), output_id

class DynamicComposerCONLLIterableDataset(CONLLIterableDatasetBase) :
    def _create_layer(self,
            phrase: conllu.models.TokenList,
            nodes: List[conllu.models.TokenTree]) -> List[Optional[conllu.models.TokenTree]] :
        layer = [None] * len(phrase)
        for node in nodes :
            layer[phrase.index(node)] = node

    def create_sample(self, np: conllu.models.TokenTree) -> Tuple[torch.LongTensor,
            List[torch.LongTensor],
            List[torch.LongTensor],
            torch.LongTensor] :
        token_list = np.to_list()

        layers = [[np]]
        while layer := self._create_layer(token_list,
                [node.children for node in layers[-1]]) :
            layers.append(layer)

        form_list = [token.get("form") for token in token_list]
        target_token = "_".join(form_list)
        if not self.token_presence_checker(target_token) :
            return None

        input_ids = torch.LongTensor([self.token_id_getter(form)
                for form in form_list])
        output_id = torch.LongTensor([self.token_id_getter(target_token)])

        # TODO use a reserved dep type id when position is not applicable to layer
        dep_ids = [torch.LongTensor([0 if node is None else self._dep_to_id[node.token.get("deprel")]
                for node in layer]) for layer in layers]

        id_to_idx = {token.get("id") : i for i, token in enumerate(token_list)}
        head_idcs = [torch.LongTensor([0 if node is None else id_to_idx.get(node.token.get("head"), 0)
                for node in layer]) for layer in layers]

        return (self.pad_fn(input_ids = input_ids)[0],
                [self.pad_fn(dep_ids = layer_dep_ids)[1] for layer_dep_ids in dep_ids],
                [self.pad_fn(head_idcs = layer_head_idcs)[2] for layer_head_idcs in head_idcs],
                output_id)

