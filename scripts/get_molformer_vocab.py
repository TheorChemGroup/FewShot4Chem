"""Write the original MoLFormer's `vocab.json` from IBM's published tokenizer.

The original MoLFormer checkpoint (config `molformer`) needs `models/weights/molformer/vocab.json`,
a {token: id} map that IBM does not publish as a separate file. It is the vocabulary of IBM's
Hugging Face tokenizer, which this script extracts from a pinned revision and checks against the
file used for the paper (2,362 tokens; content hash below, independent of key order).

    python scripts/get_molformer_vocab.py
"""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import click

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_URL = (
    "https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/resolve/"
    "361063d0ad524ef77cf39b08469f6be770dc550f/tokenizer.json"
)
# sha256 of json.dumps(vocab, sort_keys=True) for the vocab.json used for the paper.
VOCAB_CONTENT_SHA256 = "b4a5c763b877f9231af892249bec1a657aff31adb2970309ae86ece8bfce09d7"


def content_sha256(vocab):
    return hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()


@click.command()
@click.option("--out", type=click.Path(path_type=Path), default=ROOT / "models/weights/molformer/vocab.json",
              show_default=True)
def main(out):
    with urlopen(TOKENIZER_URL) as response:
        vocab = json.load(response)["model"]["vocab"]
    if content_sha256(vocab) != VOCAB_CONTENT_SHA256:
        raise click.ClickException("downloaded vocabulary differs from the one used for the paper")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, indent=4) + "\n", encoding="utf-8")
    print(f"{out} ({len(vocab)} tokens, matches the paper's vocabulary)")


if __name__ == "__main__":
    main()
