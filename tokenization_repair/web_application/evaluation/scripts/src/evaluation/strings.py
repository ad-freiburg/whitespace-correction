
def top_k_string(probs, k, encoder):
    pairs = [(probs[i], i) for i in range(len(probs))]
    top = sorted(pairs, reverse=True)[:k]
    string = ""
    for prob, label in top:
        if len(string) > 0:
            string += ' '
        char = encoder.decode_label(label)
        string += "(%s %.2f)" % (char, prob)
    return string
