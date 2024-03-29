#!/usr/bin/env python3

import unittest

import torch
from transformers import AutoTokenizer

from composer.model import Composer, DependencyEncoding

class InferenceTest(unittest.TestCase) :
    def test_end_to_end(self) :
        t = AutoTokenizer.from_pretrained("distilgpt2")
        tok = t("this is a test", return_tensors = "pt")

        c = Composer(500, 6000, 500, 20, 500, 4, 3, dtype = torch.float)
        d = DependencyEncoding(types = torch.LongTensor([[0, 1, 2, 3]]), heads = torch.LongTensor([[1, 0, 3, 1]]))

        result = c(tok.input_ids, d)
        assert result.shape == (1, 4, 500)
