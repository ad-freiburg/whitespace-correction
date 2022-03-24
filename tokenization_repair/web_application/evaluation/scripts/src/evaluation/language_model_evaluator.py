import numpy as np

from src.helper.data_structures import gather
from src.evaluation.metrics import perplexity, precision_recall_f1, crossentroy
from src.helper.data_structures import sorted_position
from src.evaluation.strings import top_k_string
from src.encoding.character_encoder import CharacterEncoder


def ground_truth_labels(prediction):
    return prediction["labels"]  # [1:-1]


def gather_probabilities(probabilities, labels):
    return gather(probabilities, labels)


def count_correct(predictions, labels):
    return np.sum(predictions == labels)


def count_tokens(sequence):
    return sequence.count(' ') + 1


class LanguageModelEvaluator:
    def __init__(self,
                 encoder: CharacterEncoder):
        self.encoder = encoder
        self.character_probabilities = []
        self.top_k_count = np.zeros(encoder.dim())
        self.n_tokens = 0
        self.spaces_tp = 0
        self.spaces_fp = 0
        self.spaces_fn = 0

    def check_space(self, predicted_label, true_label):
        space_label = self.encoder.encode_char(' ')
        if predicted_label == space_label and true_label == space_label:
            self.spaces_tp += 1
        elif predicted_label == space_label:
            self.spaces_fp += 1
        elif true_label == space_label:
            self.spaces_fn += 1

    def register_result(self, sequence, prediction):
        n_characters = len(sequence)
        character_labels = ground_truth_labels(prediction)
        probabilities = prediction["probabilities"]
        predictions = prediction["predictions"]
        character_probabilities = gather_probabilities(probabilities, character_labels)
        n_correct = count_correct(predictions, character_labels)
        n_tokens = count_tokens(sequence)
        for i, label in enumerate(character_labels):
            top_i = sorted_position(probabilities[i, :], label)
            self.top_k_count[top_i] += 1
            self.check_space(predictions[i], label)
        self.character_probabilities += list(character_probabilities)
        self.n_tokens += n_tokens
        seq_char_acc = n_correct / n_characters
        seq_crossentropy = crossentroy(character_probabilities)
        seq_char_perplexity = perplexity(character_probabilities)
        seq_token_perplexity = perplexity(character_probabilities, n=n_tokens)
        for i in range(n_characters):
            print(sequence[i], character_probabilities[i], top_k_string(probabilities[i, :], 5, self.encoder))
        print("(sequence) %.4f character accuracy" % seq_char_acc)
        print("(sequence) %.4f crossentropy" % seq_crossentropy)
        print("(sequence) %.2f character perplexity" % seq_char_perplexity)
        print("(sequence) %.2f token perplexity" % seq_token_perplexity)

    def n_chars(self):
        return len(self.character_probabilities)

    def top_k_accuracies(self, k):
        return np.cumsum(self.top_k_count[:k]) / self.n_chars()

    def print_summary(self):
        character_accuracy = self.top_k_count[0] / self.n_chars()
        top_k_accuracies = ""
        for a in self.top_k_accuracies(5):
            top_k_accuracies += "%.4f " % a
        total_crossentropy = crossentroy(self.character_probabilities)
        character_perplexity = perplexity(self.character_probabilities)
        token_perplexity = perplexity(self.character_probabilities, n=self.n_tokens)
        space_precision, space_recall, space_f1 = precision_recall_f1(self.spaces_tp, self.spaces_fp, self.spaces_fn)
        print("(total)    %.4f character accuracy" % character_accuracy)
        print("(total)    %stop-k accuracies" % top_k_accuracies)
        print("(total)    %.4f crossentropy" % total_crossentropy)
        print("(total)    %.2f character perplexity" % character_perplexity)
        print("(total)    %.2f token perplexity" % token_perplexity)
        print("(total)    %.4f space precision (%i/%i)" %
              (space_precision, self.spaces_tp, self.spaces_tp + self.spaces_fp))
        print("(total)    %.4f space recall (%i/%i)" %
              (space_recall, self.spaces_tp, self.spaces_tp + self.spaces_fn))
        print("(total)    %.4f space f1" % space_f1)
