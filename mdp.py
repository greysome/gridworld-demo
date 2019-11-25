import numpy as np

class MDP(object):
    def __init__(self, P, R, gamma):
        for P_a in P:
            for row in P_a:
                if row.sum() != 1:
                    raise ValueError('state transition probabilities for ' +
                                     'each state must add up to 1')

        self.n_actions = P.shape[0]
        self.n_states = P.shape[1]
        if R.shape[0] != self.n_states:
            raise ValueError('number of states in R and ' +
                             'P do not match')

        self.P = P
        self.R = R
        self.gamma = gamma

    def random_policy(self):
        probability = 1/self.n_actions
        return np.full((self.n_states, self.n_actions), probability)
 
    def sample(self, policy=None, state=None):
        if policy == None:
            policy = self.random_policy()
        if state == None:
            state = self._get_start_state()
        self._check_policy_probs(policy)

        fetch = 0
        samples = []
        while True:
            action = self._get_action(state, policy)
            reward = self.R[state,action]
            fetch += reward
            if self._is_terminal_state(state):
                break
            new_state = self._get_next_state(state, action)
            samples.append((state, action, reward, new_state))
            state = new_state
        return fetch, samples

    def evaluate_policy(self, policy=None, method=''):
        if policy == None:
            policy = self.random_policy()
        if method == '':
            method = ('solve' if self.n_states < 30 else 'iter')
        self._check_policy_probs(policy)

        if method not in ('solve', 'iter'):
            raise ValueError('method must be one of "solve", "iter"')

        if method == 'solve':
            yield from self._solve_policy(policy)
        elif method == 'iter':
            yield from self._iterative_policy_eval(policy)

    def value_iteration(self):
        v = np.zeros(self.n_states)

        while True:
            old_v = v
            for s in range(self.n_states):
                if self._is_terminal_state(s):
                    continue
                q_s = [self._bellman_optimality_expr(s,a,v)
                       for a in range(self.n_actions)]
                v[s] = np.max(q_s)

            epsilon = np.sum(np.abs(v-old_v))
            yield v, epsilon

    def policy_iteration(self, policy=None):
        if policy == None:
            policy = self.random_policy()
        self._check_policy_probs(policy)

        P_pi = self._get_P_pi(policy)
        R_pi = self._get_R_pi(policy)
        v = np.zeros(self.n_states)

        while True:
            old_v = v
            v = self._v_backup_synchronous(v, R_pi, P_pi)

            for s in range(self.n_states):
                if self._is_terminal_state(s):
                    continue
                q_s = [self._bellman_optimality_expr(s,a,v)
                       for a in range(self.n_actions)]

                # make the policy choose the best actions with equal
                # probability
                maximum = np.max(q_s)
                best_actions = np.argwhere(q_s == maximum)
                policy[s].fill(0)
                np.put(policy[s], best_actions, 1/best_actions.size)

            epsilon = np.sum(np.abs(v-old_v))
            yield v, policy, epsilon

    def _check_policy_probs(self, policy):
        for policy_s in policy:
            if np.sum(policy_s) != 1:
                raise ValueError('sum of probabilities for each state ' +
                                 'must be 1')

    def _is_terminal_state(self, state):
        # if the state can only transition to itself
        for P_a in self.P:
            if P_a[state,state] != 1:
                return False
        return True

    def _get_start_state(self):
        while True:
            state = np.random.randint(self.n_states)
            if not self._is_terminal_state(state):
                return state

    def _get_action(self, state, policy):
        return np.random.choice(np.arange(self.n_actions),
                                p=policy[state])

    def _get_next_state(self, state, action):
        return np.random.choice(np.arange(self.n_states),
                                p=self.P[action,state])

    def _get_R_pi(self, policy):
        R_pi = np.arange(self.n_states)
        map_R_pi = lambda s: np.sum(policy[s,a]*self.R[s,a] for a in
                            np.arange(self.n_actions))
        return map_R_pi(R_pi)

    def _get_P_pi(self, policy):
        P_pi = np.zeros((self.n_states,self.n_states))
        actions = np.arange(self.n_actions)

        for s, row in enumerate(P_pi):
            for s_, val in enumerate(row):
                P_pi[s,s_] = np.sum(policy[s,a]*self.P[a,s,s_] for a in
                                    range(self.n_actions))

        return P_pi

    def _v_backup_synchronous(self, v, R_pi, P_pi):
        return R_pi + self.gamma * np.matmul(P_pi, v)

    def _bellman_optimality_expr(self, s, a, v):
        return self.R[s,a] + self.gamma * np.sum(self.P[a,s,s_]*v[s_]
                                                 for s_ in range(self.n_states))

    def _solve_policy(self, policy):
        P_pi = self._get_P_pi(policy)
        R_pi = self._get_R_pi(policy)
        while True:
            try:
                yield np.matmul(np.linalg.inv(np.identity(self.n_states) -
                                            self.gamma*P_pi),
                                R_pi), 0
            except np.linalg.LinAlgError:
                yield from self._iterative_policy_eval(policy)

    def _iterative_policy_eval(self, policy):
        P_pi = self._get_P_pi(policy)
        R_pi = self._get_R_pi(policy)
        v = np.zeros(self.n_states)

        while True:
            old_v = v
            v = self._v_backup_synchronous(v, R_pi, P_pi)
            epsilon = np.sum(np.abs(v-old_v))
            yield v, epsilon
