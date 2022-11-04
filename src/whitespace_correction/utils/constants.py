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
UNK_LANG = "[unk]"
EN = "[en]"
DE = "[de]"
ES = "[es]"
FR = "[fr]"
IT = "[it]"
PR = "[pr]"

LANGUAGE_TOKENS = [UNK_LANG, EN, DE, ES, FR, IT, PR]
