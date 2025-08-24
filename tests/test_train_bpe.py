import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode

def test_train_bpe_is_deterministic():
    input_path = FIXTURES_PATH / "corpus.en"
    args = dict(input_path=input_path, num_merges=10, special_tokens=["<|endoftext|>"])
    v1, m1 = run_train_bpe(**args)
    v2, m2 = run_train_bpe(**args)
    assert m1 == m2
    assert v1 == v2


def test_zero_merges():
    input_path = FIXTURES_PATH / "corpus.en"
    _, merges = run_train_bpe(input_path=input_path, num_merges=0, special_tokens=["<|endoftext|>"])
    assert merges == []


def test_unicode_words(tmp_path):
    p = tmp_path / "pt.txt"
    p.write_text("coração ação informação", encoding="utf-8")
    vocab, merges = run_train_bpe(input_path=p, num_merges=50, special_tokens=["<|endoftext|>"])

    # Todos os valores do vocab são bytes e todos decodificam como UTF-8
    assert all(isinstance(b, (bytes, bytearray)) for b in vocab.values())
    for b in vocab.values():
        b.decode("utf-8")  # falha o teste se algum não decodificar

    # merges tem a estrutura esperada: lista de pares de bytes
    assert isinstance(merges, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in merges)
    assert all(isinstance(p[0], (bytes, bytearray)) and isinstance(p[1], (bytes, bytearray)) for p in merges)


def test_vocab_grows_with_more_merges():
    """
    O vocabulário deve (em geral) aumentar quando treinamos com mais merges.
    Não exigimos supersete exato (porque representações podem mudar),
    mas o tamanho deve crescer ou, no mínimo, não diminuir.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    special = ["<|endoftext|>"]

    vocab_small, merges_small = run_train_bpe(
        input_path=input_path, num_merges=50, special_tokens=special
    )
    vocab_big, merges_big = run_train_bpe(
        input_path=input_path, num_merges=200, special_tokens=special
    )

    # Mais merges → nunca menos merges executados
    assert len(merges_big) >= len(merges_small)

    # Mais merges → vocabulário tende a crescer (mais unidades aprendidas)
    assert len(vocab_big) >= len(vocab_small)


def test_train_bpe_speed():
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        num_merges=50,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 25
