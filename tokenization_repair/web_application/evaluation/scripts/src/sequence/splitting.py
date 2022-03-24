from src.sequence.transformation import get_word_positions

from numpy import cumsum, sum


def merge(tokens):
    spaces = list(cumsum([len(t) for t in tokens[:-1]]))
    merged = "".join(tokens)
    is_space = [False] * len(merged)
    for pos in spaces:
        is_space[pos] = True
    return merged, is_space


def word_beginnings_array(word_positions, seqlen):
    array = [[] for _ in range(seqlen + 1)]
    for pos in word_positions:
        array[pos[1]].append(pos[0])
    return array


class PartialSolution:
    def __init__(self, predecessors, nonwords, operations):
        self.predecessors = predecessors
        self.nonwords = nonwords
        self.operations = operations

    def __str__(self):
        return "predecessors=%s, nonwords=%i, operations=%i" %(str(self.predecessors), self.nonwords, self.operations)

    def __repr__(self):
        return self.__str__()


def append_backward_steps(current, best_partial_solutions):
    predecessors = best_partial_solutions[current].predecessors
    if predecessors is None:
        return [[current]]
    backward_steps = []
    for i in range(len(predecessors)):
        solutions = append_backward_steps(predecessors[i], best_partial_solutions)
        for s in solutions:
            backward_steps.append(s + [current])
    return backward_steps


def recover_best_splits(best_partial_solutions):
    backward_splits = append_backward_steps(len(best_partial_solutions) - 1, best_partial_solutions)
    return backward_splits


def dynamic_best_splits(merged, spaces, word_positions):
    beginnings_array = word_beginnings_array(word_positions, len(merged))
    #for i in range(len(beginnings_array)):
    #    print(i, beginnings_array[i])
    best_partial_solutions = [PartialSolution(None, 0, 0)]
    last_nonword_begin = 0
    for i in range(1, len(merged) + 1):
        # append nonword:
        predecessor = last_nonword_begin
        #if i < len(spaces) and spaces[i]:
        #    last_nonword_begin = i
        best_predecessors = [predecessor]
        best_nonwords = best_partial_solutions[predecessor].nonwords + 1
        best_operations = best_partial_solutions[predecessor].operations + sum(spaces[(predecessor+1):i], dtype=int)
        nonword_is_best = True
        #print(best_predecessor, best_nonwords, best_operations)
        # append words ending at current position:
        for word_beginning in beginnings_array[i]:
            nonwords = best_partial_solutions[word_beginning].nonwords
            operations = best_partial_solutions[word_beginning].operations
            operations += 0 if word_beginning == 0 or spaces[word_beginning] else 1
            operations += sum(spaces[(word_beginning+1):i], dtype=int)
            #print(word_beginning, nonwords, operations)
            if nonwords < best_nonwords or (nonwords == best_nonwords and operations < best_operations):
                best_predecessors = [word_beginning]
                best_nonwords = nonwords
                best_operations = operations
                nonword_is_best = False
            elif nonwords == best_nonwords and operations == best_operations:
                best_predecessors.append(word_beginning)
        if not nonword_is_best and i < len(spaces) and spaces[i]:
            last_nonword_begin = i
        """if i < len(spaces) and spaces[i]:
            next_space = None
            for j in range(i, len(merged)):
                if spaces[j]:
                    next_space = j
                    break
            if next_space is not None and i not in beginnings_array[next_space]:
                last_nonword_begin = i"""
        partial_solution = PartialSolution(best_predecessors, best_nonwords, best_operations)
        best_partial_solutions.append(partial_solution)
        #print("pos=%i, %s" % (i, partial_solution))
    best_splits = recover_best_splits(best_partial_solutions)
    return best_splits


def get_best_splits(tokens, word_counters):
    merged, spaces = merge(tokens)
    word_positions = get_word_positions(merged, word_counters)
    split_position_array = dynamic_best_splits(merged, spaces, word_positions)
    splits = []
    for split_positions in split_position_array:
        split = []
        for i in range(len(split_positions) - 1):
            split.append((split_positions[i], split_positions[i + 1]))
        splits.append(split)
    return splits
