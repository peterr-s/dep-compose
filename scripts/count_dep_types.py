#!/usr/bin/env python3

import sys
import logging

import conllu

from composer.utils import configure_logging

log = logging.getLogger(__name__)
configure_logging()

def main() :
    dep_types = {token.get("deprel")
            for parse in conllu.parse_incr(sys.stdin)
            for token in parse}

    log.info(f"Found {len(dep_types)} dependency types.")
    log.debug(f"{dep_types=}")

if __name__ == "__main__" :
    main()
