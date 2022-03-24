class OCRTokenCorruptor:
    @staticmethod
    def corrupt(sequence):
        corrupt_sequence = ""
        for i in range(len(sequence)):
            if i > 0 and sequence[i -1 ] != ' ' and sequence[i] != ' ':
                corrupt_sequence += ' '
            corrupt_sequence += sequence[i]
        return corrupt_sequence
