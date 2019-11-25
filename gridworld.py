import numpy as np
from mdp import MDP

class GridWorld(object):
    def __init__(self, w, h, terminal_states):
        self.n_actions = 4

        if len(terminal_states) < 1:
            raise ValueError('there must be at least 1 terminal state')
        self.terminal_states = terminal_states

        self.set_wh(w, h)

    def set_w(self, w):
        self.set_wh(w, self.h)

    def set_h(self, h):
        self.set_wh(self.w, h)

    def set_wh(self, w, h):
        self.terminal_states = [0]
        self.w, self.h = w, h
        self.n_states = w*h

        self._build_all()
        self._build_MDP()

    def set_transition_probs(self, s, P_s_a, a=None):
        # `P_s_a` is a `n_states` vector representing the probability
        # of transitioning to each state, given state `s`

        # if action is None, set transition probabilities as `P_s_a`
        # for all actions in state `s`
        # otherwise, set transition probabilities as `P_s_a` only for
        # action `a` in state `s`

        if s in self.terminal_states:
            raise ValueError('not allowed to set state transitions ' +
                             'of terminal state')
        self._verify_probs(P_s_a)

        if a == None:
            for a in range(self.n_actions):
                self.P[a,s] = P_s_a
        else:
            for i, s_ in enumerate(S_):
                self.P[a,s] = P_s_a

    def set_reward(self, s, r, a=None):
        # if action is None, set reward as `r` for all actions in state `s`
        # otherwise, set reward as `r` only for action `a` in state `s`
        if s in self.terminal_states:
            raise ValueError('not allowed to set reward of terminal state')
        if a == None:
            self.R[s].fill(r)
        else:
            self.R[s,a] = r

    def toggle_terminal_state(self, s):
        if s in self.terminal_states:
            self.terminal_states.remove(s)
            if len(self.terminal_states) < 1:
                raise ValueError('there must be at least 1 terminal state')
            self._build_state(s, terminal=False, default=True)
        else:
            self.terminal_states.append(s)
            for a in range(4):
                self.P[a,s].fill(0)
                self.P[a,s,s] = 1
            self.R[s].fill(0)

    def pretty_print_values(self, v):
        for i in range(self.h):
            for j in range(self.w):
                print('{:.2f}'.format(v[i*self.w+j]).ljust(7),
                      end=' ')
            print()

    def pretty_print_policy(self, policy):
        char_map = {0: 'l', 1: 'r', 2: 'u', 3: 'd'}
        for i in range(self.h):
            for j in range(self.w):
                if i*self.w+j in self.terminal_states:
                    print('T'.ljust(4), end=' ')
                else:
                    best_actions = self._best_actions(policy[i*self.w+j])
                    print(''.join(char_map[a] for a in best_actions).ljust(4),
                          end=' ')
            print()

    def _next_state(self, a, s):
        return (s if self.no_offset_map[a](s) else s + self.offset_map[a])

    def _build_state(self, s, terminal, default=False,
                     P_s=None, R_s=None):
        # set state transition probabilities and rewards for state `s`.
        # if state is non-terminal and default is False,
        # `P_s` and `R_s` have to be provided

        # `R_s` is a `n_action`-vector representing the expected reward
        # of each action

        # `P_s` is a `n_actions`*`n_states` matrix containing the
        # probabilities of transitioning to every state for each
        # action

        if terminal:
            for a in range(4):
                self.P[a,s].fill(0)
                self.P[a,s,s] = 1
            self.R[s].fill(0)
        else:
            if default:
                R_s = np.full(4,-1)
                P_s = self._default_state_transitions(s)

            for a in range(self.n_actions):
                self._verify_probs(P_s[a])
                self.P[a,s] = P_s[a]
            self.R[s] = R_s

    def _build_all(self):
        # this must be called first on initialisation, or when
        # dimension are changed
        self._build_offset_maps()
        self.P = np.zeros((self.n_actions, self.n_states, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            if s in self.terminal_states:
                self._build_state(s, terminal=True)
            else:
                self._build_state(s, terminal=False, default=True)

    def _build_MDP(self):
        self.mdp = MDP(P=self.P, R=self.R, gamma=1)

    def _build_offset_maps(self):
        self.offset_map = {0: -1, 1: 1, 2: -self.w, 3: self.w}
        # when lambda s is true, the state to transition to is
        # the same as the current, i.e. the agent does not move
        self.no_offset_map = {0: lambda s: s % self.w == 0,
                              1: lambda s: s % self.w == self.w-1,
                              2: lambda s: s < self.w,
                              3: lambda s: s >= self.w*(self.h-1)}

    def _default_state_transitions(self, s):
        P_s = np.full((self.n_actions, self.n_states), 0)
        for a in range(self.n_actions):
            s_ = self._next_state(a, s)
            P_s[a,s_] = 1
        return P_s

    def _best_actions(self, policy_row):
        # np.argmax but able to return multiple values
        best_actions = np.argwhere(policy_row == np.max(policy_row)) \
                         .reshape(-1)
        return best_actions

    def _verify_probs(self, probs):
        if np.sum(probs) != 1:
            raise ValueError('sum of probabilities must be 1')
