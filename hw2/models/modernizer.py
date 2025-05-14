# SYSTEM IMPORTS
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Tuple, Type, Union
import collections
import numpy as np
import os
import sys
from tqdm import trange
from tqdm import tqdm


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from eval.cer import cer
from topological_sort_fst import fst_topological_sort
from fst import FST, EPSILON, Transition, compose, create_seq_fst, StateType
from lm import make_ngram
from vocab import END_TOKEN
from from_file import load_parallel, load_mono


# TYPES DECLARED IN THIS MODULE
ModernizerType: Type = Type["Modernizer"]



class Modernizer(object):
    def __init__(self: ModernizerType):
        self.lm: FST = None      # lanuage model
        self.tm: FST = None      # typo model

    def _create_typo_model(self: ModernizerType,
                           s_data: Sequence[str],
                           w_data: Sequence[str]
                           ) -> ModernizerType:
        #setting up the states
        self.tm = FST()  
        self.tm.set_start("q_0")  
        self.tm.add_state("q_1")
        self.tm.set_accept("q_2") 

        #setting up vocab for input and output
        source_chars = set()
        target_chars = set()
        for line in s_data:
            source_chars.update(line)
        for line in w_data:
            target_chars.update(line)

        #character to character transitions
        for s_char in source_chars:
            for w_char in target_chars:
                # Same character transition
                t_0 = Transition("q_0", (s_char, w_char), "q_0")
                self.tm.add_transition(t_0)
                t_1 = Transition("q_1",(s_char, w_char),"q_0")
                self.tm.add_transition(t_1)
        #insertion and deletion transitions
        for w_char in target_chars:
            t_insert = Transition("q_0", (EPSILON, w_char), "q_0")
            self.tm.add_transition(t_insert)
        for s_char in source_chars:
            t_delete = Transition("q_0", (s_char, EPSILON), "q_1")
            self.tm.add_transition(t_delete)
        
        #stop token transitions
        t_end_0 = Transition("q_0", (END_TOKEN, END_TOKEN), "q_2")
        self.tm.add_transition(t_end_0)
        t_end_1 = Transition("q_1", (END_TOKEN, END_TOKEN), "q_2")
        self.tm.add_transition(t_end_1)
        
    def train_language_model(self: ModernizerType,
                             data: Sequence[str]
                             ) -> ModernizerType:
        self.lm = make_ngram(data, 2) # make a bigram language model...feel free to change this if you want
        return self

    def init_typo_model(self: ModernizerType,
                        s_data: Sequence[str],
                        w_data: Sequence[str]
                        ) -> ModernizerType:
        # TODO: implement me!
        # This method should, if self.tm is None, create the FST object from the data.
        # Then you should initialize your typo model's weights. Remember that Shakespearean spelling isn't too far
        # off from what our modern spelling is, so you should (heavily) favor transforming a character into the
        # same character. you should have a small weight of transforming a character into any other
        # (non-EPSILON) character, and an even smaller weight for EPSILON transitions (remember there are two kinds
        # of EPSILON transitions and we need to consider BOTH of them!).
        if self.tm is None:
            self._create_typo_model(s_data, w_data)

        for q in self.tm.states:
            for transition in self.tm.transitions_from[q]:
                if transition.a[0] == transition.a[1]:
                    self.tm.reweight_transition(transition, wt=50.0)
                elif transition.a[0] == EPSILON or transition.a[1] == EPSILON:
                    self.tm.reweight_transition(transition, wt=0.05)
                else:
                    self.tm.reweight_transition(transition, wt=1.0)
        # 3) normalize to form conditional distributions p(output | input)
        # Also remember to think in terms of *weights*, not probabilities here. You can always call
        # self.tm.normalize_cond() to form the pmfs.
        self.tm.normalize_cond()
        return self
        
    

    def viterbi_traverse(self: ModernizerType,
                         fst: FST,
                         update_func_ptr: Callable[[StateType,  # the current state 
                                                    Transition, # incoming transition to state
                                                    float],     # weight of transition
                                                   None],        # function returns nothing
                         reverse = False
                        ) -> None:
        # TODO: implement me!
        topo_order = fst_topological_sort(fst)
        if reverse:
            topo_order.reverse()
            for state in topo_order:
                for trans, weight in fst.transitions_from[state].items():
                    update_func_ptr(state, trans, weight)
        else:
            for state in topo_order:
                for trans, weight in fst.transitions_to[state].items():
                    update_func_ptr(state, trans, weight)
        # This method should implement the viterbi traversal that is common to any flavor of the viterbi algorithm.
        # Whenever you arrive at a point in your traversal where flavors of viterbi differ, you should call
        # the update_func_ptr with the current state of the traversal.

        # The current state of a viterbi traversal is:
        #   (the state in the FST, an incoming transition to that state, the weight of that transition)
        # Note that these are the arguments that the update_func_ptr should take, and it should return nothing

    def viterbi_decode(self: ModernizerType,
                   fst: FST
                   ) -> Tuple[Sequence[StateType], float, Sequence[Transition]]:
        viterbi_score = {s: -np.inf for s in fst.states}
        parent = {}
        best_incoming = {}

        viterbi_score[fst.start] = 0.0

        def update_func(next_state, transition, weight):
            prev_state = transition.q
            candidate = viterbi_score[prev_state] + np.log(weight)
            if candidate > viterbi_score[next_state]:
                viterbi_score[next_state] = candidate
                parent[next_state] = prev_state
                best_incoming[next_state] = transition

        self.viterbi_traverse(fst, update_func)

        if viterbi_score[fst.accept] == -np.inf:
            return [], -np.inf, []

        state_path = []
        transition_path = []
        state = fst.accept

        while state != fst.start:
            state_path.append(state)
            transition_path.append(best_incoming[state])
            state = parent[state]
        

        return state_path, viterbi_score[fst.accept], transition_path


    def decode_seq(self: ModernizerType,
               lm_tm_fst: FST,
               w_seq: str
               ) -> Tuple[str, float]:
        M_seq = create_seq_fst(w_seq)
        composed = compose(lm_tm_fst, M_seq)

        _ , logprob, transition_path = self.viterbi_decode(composed)

        if logprob == -np.inf or not transition_path:
            return "", -np.inf

        transition_path.reverse()

        output = [t.a[1] for t in transition_path if t.a[1] != EPSILON and t.a[1] != END_TOKEN]
        return "".join(output), logprob


    def decode(self: ModernizerType,
           w_data: Sequence[str]
           ) -> Iterable[Tuple[str, float]]:
        lm_tm = compose(self.lm, self.tm)

        for w_seq in w_data:
            yield self.decode_seq(lm_tm, w_seq)


    def viterbi_forward(self: ModernizerType,
                        fst: FST
                        ) -> Tuple[Mapping[StateType, float]]:

        # TODO: implement me!
        # This method should use your viterbi_traverse method to implement the "forward" algorithm (which we are calling
        # viterbi_forward. Remember, the forward algorithm uses viterbi traversal to calculate the sum of all path
        # weights from the start state to any other state in the graph. These values should be returned as a mapping
        # where the key of the mapping is the fst state, and the value is the sum of path weights (i.e. probabilities)

        # Reminder, the value of the accept state from the forward algorithm should be the same as the value
        # of the start state from the backward algorithm
        
        vals = {state: 0.0 for state in fst.states}
        vals[fst.start] = 1.0

        def update_func(state, transition, weight):
            prev_state = transition.q
            vals[state] += vals[prev_state] * weight

        self.viterbi_traverse(fst, update_func)

        return vals

    def loglikelihood(self: ModernizerType,
                  w_data: Sequence[str]
                  ) -> float:
        if self.lm is None or self.tm is None:
            raise ValueError("Language model or Typo model is not initialized.")

        lm_tm_fst = compose(self.lm, self.tm)
        total_logprob = 0.0

        for w_seq in w_data:
            M_seq = create_seq_fst(w_seq)
            composed = compose(lm_tm_fst, M_seq)
            forward_scores = self.viterbi_forward(composed)
            prob = forward_scores.get(composed.accept, 0.0)

            if prob > 0.0:
                total_logprob += np.log(prob)
            else:
                total_logprob += -np.inf

        return total_logprob

    

    def viterbi_backward(self: ModernizerType,
                         fst: FST
                         ) -> Tuple[Mapping[StateType, float]]:

        # TODO: implement me!
        # This method should use your viterbi_traverse method to implement the "backward" algorithm (which we are calling
        # viterbi_backward. Remember, the backward algorithm uses viterbi traversal to calculate the sum of all path
        # weights from any state in the graph TO the accept state. These values should be returned as a mapping
        # where the key of the mapping is the fst state, and the value is the sum of path weights (i.e. probabilities)

        # Reminder, the value of the accept state from the forward algorithm should be the same as the value
        # of the start state from the backward algorithm
        vals = {state: 0.0 for state in fst.states}
        vals[fst.accept] = 1.0  # accept state starts with probability 1

        def update_func(state, transition, weight):
            next_state = transition.r
            vals[state] += vals[next_state] * weight

        self.viterbi_traverse(fst, update_func, reverse=True)
        return vals
    

    def brittle_estep(self: ModernizerType,
                    train_s_data: Sequence[str],
                    train_w_data: Sequence[str]
                    ) -> Mapping[Tuple[str, str], float]:
        counts = collections.defaultdict(float)
        for s_seq, w_seq in zip(train_s_data, train_w_data):
            Ms = create_seq_fst(s_seq)
            Mw = create_seq_fst(w_seq)
            composed_fst = compose(compose(Ms, self.tm), Mw)
            _, _, transitions = self.viterbi_decode(composed_fst)
            for t in transitions:
                s_token, w_token = t.a
                counts[(s_token, w_token)] += 1.0
        return counts
    
    def flexible_estep(self: ModernizerType,
                   train_s_data: Sequence[str],
                   train_w_data: Sequence[str]
                   ) -> Mapping[Tuple[str, str], float]:
        # TODO: implement me!
        # This method is where your soft-EM E-step should go. This method should produce a mapping of counts
        # where the key is (s_token, w_token) and the value is the "soft"-count for the number of times
        # w_token was corrected to s_token across the entire training data

        # you will have to use the forward/backward algorithm in this!

        counts = collections.defaultdict(float)

        for s_seq, w_seq in zip(train_s_data, train_w_data):
            Ms = create_seq_fst(s_seq)
            Mw = create_seq_fst(w_seq)
            composed_fst = compose(compose(Ms, self.tm), Mw)

            forward_probs = self.viterbi_forward(composed_fst)
            backward_probs = self.viterbi_backward(composed_fst)

            Z = forward_probs.get(composed_fst.accept)

            assert np.isclose(Z, backward_probs.get(composed_fst.start, 0.0), rtol=1e-12), "forward and backward mismatch"

            for state in composed_fst.states:
                for t, weight in composed_fst.transitions_from[state].items():
                    s_token, w_token = t.a
                    if s_token == EPSILON and w_token == EPSILON:
                        continue  # skip epsilon-epsilon

                    from_state = t.q
                    to_state = t.r

                    contrib = forward_probs[from_state] * weight * backward_probs[to_state]
                    counts[(s_token, w_token)] += contrib / Z if Z > 0 else 0.0

        return counts
    
    def parallel_loglikelihood(self: ModernizerType,
                               s_data: Sequence[str],
                               w_data: Sequence[str]
                               ) -> float:

        total_logprob: float = 0
        for s_seq, w_seq in tqdm(zip(s_data, w_data),
                                 total=len(w_data),
                                 desc="parallel log likelihood:"):

            s_fst: FST = create_seq_fst(s_seq)
            w_fst: FST = create_seq_fst(w_seq)
            m: FST = compose(compose(s_fst, self.tm), w_fst)

            total_logprob += np.log(self.viterbi_forward(m)[m.accept])

        return total_logprob
    

    def mstep(self: ModernizerType,
              counts: Mapping[Tuple[str, str], float],
              delta: float = 0
              ) -> None:
        # relative frequency estimation:
        #   1) use the counts to reweight transitions
        #   2) normalize the weights (can do add-delta smoothing here if you want)

        for q in self.tm.states:
            for t in self.tm.transitions_from[q].keys():
                self.tm.reweight_transition(t, wt=counts[t.a])
        self.tm.normalize_cond(add=delta)

    def brittle_train(self: ModernizerType,
                      train_s_data: Sequence[str],
                      train_w_data: Sequence[str],
                      test_data: Tuple[Sequence[str], Sequence[str]] = None,
                      delta: float = 0,
                      max_iters: int = 1000,
                      converge_error: float = 1e-5,
                      loglikelihoods: list[float] = None
                      ) -> ModernizerType:

        current_iter: int = 0
        current_logprob: float = None
        prev_logprob: float = -np.inf # initial error for logprob
        percent_rel_error: float = 1.0
        # while we haven't converged and we haven't given up
        while percent_rel_error >= converge_error and current_iter < max_iters:
            # estep
            counts: Mapping[Tuple[str, str], float] = self.brittle_estep(train_s_data, train_w_data)
            # mstep
            self.mstep(counts, delta=delta)
            # now evaluate error
            prev_logprob = current_logprob
            current_logprob = self.parallel_loglikelihood(train_s_data, train_w_data)
            if loglikelihoods is not None:
                loglikelihoods.append(current_logprob)
            if prev_logprob is None:
                percent_rel_error = 1
            else:
                percent_rel_error = abs(prev_logprob - current_logprob) / abs(prev_logprob)
            current_iter += 1
            print(f"after iter={current_iter} logprob={current_logprob}")

            if test_data is not None:
                # eval on test data if it is present as a function of EM iteration
                print("test data is not None....evaluating")
                test_s_data, test_w_data = test_data

                test_s_predictions: Sequence[str] = list()
                for idx, (s_predicted_seq, logprob) in enumerate(self.decode(test_w_data)):
                    if idx < 10:
                        print(f"{s_predicted_seq}\t{logprob}")
                    test_s_predictions.append(s_predicted_seq)

                score: float = cer(zip(test_s_data, test_s_predictions))
                print(f"after iter={current_iter} cer={score}")

        return self

    def flexible_train(self: ModernizerType,
                       train_s_data: Sequence[str],
                       train_w_data: Sequence[str],
                       test_data: Tuple[Sequence[str], Sequence[str]] = None,
                       delta: float = 0,
                       max_iters: int = 1000,
                       converge_error: float = 1e-5,
                       loglikelihoods: list[float] = None
                       ) -> ModernizerType:

        current_iter: int = 0
        current_logprob: float = None
        prev_logprob: float = -np.inf # initial error for logprob
        percent_rel_error: float = 1.0

        # while we haven't converged and we haven't given up
        while percent_rel_error >= converge_error and current_iter < max_iters:

            # estep
            counts: Mapping[Tuple[str, str], float] = self.flexible_estep(train_s_data, train_w_data)

            # mstep
            self.mstep(counts, delta=delta)

            # now evaluate error
            prev_logprob = current_logprob
            current_logprob = self.parallel_loglikelihood(train_s_data, train_w_data)
            if loglikelihoods is not None:
                loglikelihoods.append(current_logprob)
            if prev_logprob is None:
                percent_rel_error = 1
            else:
                percent_rel_error = abs(prev_logprob - current_logprob) / abs(prev_logprob)
            current_iter += 1

            print(f"after iter={current_iter} logprob={current_logprob}")

            if test_data is not None:
                # eval on test data if it is present as a function of EM iteration
                print("test data is not None....evaluating")
                test_s_data, test_w_data = test_data

                test_s_predictions: Sequence[str] = list()
                for idx, (s_predicted_seq, logprob) in enumerate(self.decode(test_w_data)):
                    if idx < 10:
                        print(f"{s_predicted_seq}\t{logprob}")
                    test_s_predictions.append(s_predicted_seq)

                score: float = cer(zip(test_s_data, test_s_predictions))
                print(f"after iter={current_iter} cer={score}")

        return self
    def visualize_lm(self):
        if self.lm is not None:
            self.lm.visualize()

    def visualize_tm(self):
        if self.tm is not None:
            self.tm.visualize()

