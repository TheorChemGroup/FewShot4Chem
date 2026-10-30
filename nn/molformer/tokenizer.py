import json

import regex as re
from transformers import BertTokenizer

PATTERN = "(\\[[^\\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\\(|\\)|\\.|=|#|-|\\+|\\\\|\\/|:|~|@|\\?|>|\\*|\\$|\\%[0-9]{2}|[0-9])"


class MolTranBertTokenizer(BertTokenizer):
    def __init__(
        self,
        vocab_file: str = "",
        do_lower_case=False,
        unk_token="<pad>",
        sep_token="<eos>",
        pad_token="<pad>",
        cls_token="<bos>",
        mask_token="<mask>",
        **kwargs,
    ) -> None:
        super().__init__(
            vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # BertTokenizer reads vocab_file line-by-line (text format), but IBM MoLFormer
        # ships vocab.json as a {token: id} dict. Re-load it correctly here.
        if vocab_file and vocab_file.endswith(".json"):
            with open(vocab_file, encoding="utf-8") as f:
                self.vocab = json.load(f)
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        self.regex_tokenizer = re.compile(PATTERN)
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string
