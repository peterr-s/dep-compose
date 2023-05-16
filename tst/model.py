import unittest
import logging

import torch

from composer.model import *
from composer.utils import configure_logging

log = logging.getLogger(__name__)
configure_logging()

class UtilityTest(unittest.TestCase) :
    def test_generate_mask_happy_path(self) :
        heads = torch.LongTensor([[1, 0, 1, 2]])
        embedding_dim = 5
        mask = generate_mask(heads, embedding_dim)
        expected = torch.BoolTensor(
            [[[[False, False, False, False, False],
              [ True,  True,  True,  True,  True],
              [False, False, False, False, False],
              [False, False, False, False, False]],
             [[ True,  True,  True,  True,  True],
              [False, False, False, False, False],
              [ True,  True,  True,  True,  True],
              [False, False, False, False, False]],
             [[False, False, False, False, False],
              [False, False, False, False, False],
              [False, False, False, False, False],
              [ True,  True,  True,  True,  True]],
             [[False, False, False, False, False],
              [False, False, False, False, False],
              [False, False, False, False, False],
              [False, False, False, False, False]]]])

        assert torch.equal(mask, expected)

    def test_generate_mask_empty_heads(self) :
        pass

class CompositionBlockTest(unittest.TestCase) :
    def test_get_device(self) :
        if torch.cuda.is_available() :
            composition_block = CompositionBlock()

            composition_block.to("cpu")
            assert composition_block.device() == torch.device("cpu")
            composition_block.to("cuda:0")
            assert composition_block.device() == torch.device("cuda:0")
        else :
            log.warn("CUDA is not available; can't test moving model between devices")

class Composer(unittest.TestCase) :
    pass
