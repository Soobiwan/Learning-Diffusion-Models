from __future__ import annotations

import pytest
from tests.helpers import DummyTokenizer


@pytest.fixture()
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()
