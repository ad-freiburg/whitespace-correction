# tokens for marking elements of text that are not regular language tokens
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
SEP = "<sep>"
UNK = "<unk>"
MASK = "<mask>"

SPECIAL_TOKENS = [UNK, BOS, EOS, PAD]

# sep and mask are only used for special purposes, such as decoder only models or masked language modeling
EXTENDED_SPECIAL_TOKENS = SPECIAL_TOKENS + [SEP, MASK]

# tokens for marking the language of a text (including a unknown language token)
LANGUAGE_CODES = ["unk", "en", "de", "es", "fr", "it", "pt"]
LANGUAGE_CODES_TO_TOKENS = {
    lc: f"[{lc}]"
    for lc in LANGUAGE_CODES
}
LANGUAGE_TOKENS = [LANGUAGE_CODES_TO_TOKENS[lc] for lc in LANGUAGE_CODES]
