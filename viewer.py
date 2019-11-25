from time import sleep
from threading import Thread
import tkinter as tk
import numpy as np

class GridWorldViewer(object):
    def __init__(self, gw):
        self.gw = gw
        self.running = True
        self.mainloop_done = False

        self.delay = 0.5
        self.pause = False
        self.view = 'values'
        self.cur_values = None
        self.cur_policy = None
        self.cur_rewards = None

        self.canvas_w = self.gw.w * 100
        self.canvas_h = self.gw.h * 100
        self.cv = tk.Canvas(width=self.canvas_w, height=self.canvas_h)
        self.cv.master.bind('v', lambda event: setattr(self, 'view', 'values'))
        self.cv.master.bind('p', lambda event: setattr(self, 'view', 'policy'))
        self.cv.master.bind('r', lambda event: setattr(self, 'view', 'rewards'))
        self.cv.master.bind(']', self.faster)
        self.cv.master.bind('[', self.slower)
        self.cv.master.bind('-', self.toggle_pause)
        self.cv.master.bind('q', self.quit)
        self.cv.pack()

        self.thread = Thread(target=self.collect_data)

    def run(self):
        try:
            self.thread.start()
            self.mainloop()
        except KeyboardInterrupt:
            self.quit(None)

    def faster(self, event):
        old_delay = self.delay
        self.delay /= 2
        if self.delay < 0:
            self.delay = old_delay

    def slower(self, event):
        old_delay = self.delay
        self.delay *= 2
        if self.delay > 2:
            self.delay = old_delay

    def quit(self, event):
        self.running = False
        self.thread.join()
        self.cv.quit()
        self.cv.master.destroy()

    def toggle_pause(self, event):
        self.pause = not self.pause

    def collect_data(self):
        # this will be run in the background
        while self.running:
            if self.pause:
                # prevents the program from slowing down
                sleep(self.delay)
                continue
            self.cur_values = self.get_values()
            self.cur_policy = self.get_policy()
            self.cur_rewards = self.get_rewards()
            sleep(self.delay)

    def mainloop(self):
        if self.running:
            self.cv.delete('all')
            if self.view == 'values':
                self.update_values_view(self.cur_values)
            elif self.view == 'policy':
                self.update_policy_view(self.cur_policy)
            elif self.view == 'rewards':
                self.update_rewards_view(self.cur_rewards)
            self.cv.update()
        self.cv.master.after(1, self.mainloop)

    def get_values(self):
        pass

    def get_policy(self):
        pass

    def get_rewards(self):
        pass

    def coords_generator(self):
        w_r = self.canvas_w / self.gw.w
        h_r = self.canvas_h / self.gw.h

        for i in range(self.gw.h):
            for j in range(self.gw.w):
                yield w_r, h_r, i, j

    def update_values_view(self, values):
        # TODO: colours
        for w_r, h_r, i, j in self.coords_generator():
            if i*self.gw.w+j in self.gw.terminal_states:
                self.cv.create_rectangle(j*w_r, i*h_r,
                                            (j+1)*w_r, (i+1)*h_r,
                                            fill='grey')
                continue

            self.cv.create_rectangle(j*w_r, i*h_r,
                                     (j+1)*w_r, (i+1)*h_r,
                                     fill='white')
            if self.cur_values is None:
                continue

            text = '{:.2f}'.format(self.cur_values[i*self.gw.w+j])
            self.cv.create_text(((j+0.5)*w_r, (i+0.5)*h_r),
                                text=text,
                                font=('Arial', 20))

    def update_policy_view(self, policy):
        end_coords = lambda i,j: {0: ((j+0.3)*w_r, (i+0.5)*h_r),
                             1: ((j+0.7)*w_r, (i+0.5)*h_r),
                             2: ((j+0.5)*w_r, (i+0.3)*h_r),
                             3: ((j+0.5)*w_r, (i+0.7)*h_r)}

        if policy is None:
            return
        
        for w_r, h_r, i, j in self.coords_generator():
            for a in policy[i*self.gw.w+j]:
                if i*self.gw.w+j in self.gw.terminal_states:
                    self.cv.create_rectangle(j*w_r, i*h_r,
                                             (j+1)*w_r, (i+1)*h_r,
                                             fill='grey')
                    continue

                self.cv.create_rectangle(j*w_r, i*h_r,
                                         (j+1)*w_r, (i+1)*h_r,
                                         fill='white')
                self.cv.create_line(((j+0.5)*w_r, (i+0.5)*h_r),
                                    end_coords(i,j)[a],
                                    arrow = tk.LAST)

    def update_rewards_view(self, policy):
        text_coords = lambda i,j: {0: ((j+0.25)*w_r, (i+0.5)*h_r),
                              1: ((j+0.75)*w_r, (i+0.5)*h_r),
                              2: ((j+0.5)*w_r, (i+0.25)*h_r),
                              3: ((j+0.5)*w_r, (i+0.75)*h_r)}
        for w_r, h_r, i, j in self.coords_generator():
            if i*self.gw.w+j in self.gw.terminal_states:
                self.cv.create_rectangle(j*w_r, i*h_r,
                                         (j+1)*w_r, (i+1)*h_r,
                                         fill='grey')
                continue

            self.cv.create_rectangle(j*w_r, i*h_r,
                                     (j+1)*w_r, (i+1)*h_r,
                                     fill='white')

            if self.cur_rewards is None:
                continue

            for a in range(self.gw.n_actions):
                text = '{:.2f}'.format(self.cur_rewards[i*self.gw.w+j,a])
                self.cv.create_text(text_coords(i,j)[a],
                                    text=text,
                                    font=('Arial', 15))

class ValueIterationViewer(GridWorldViewer):
    def __init__(self, gw):
        super().__init__(gw)
        self.values_iter = gw.mdp.value_iteration()
        self.last_values = None

    def get_values(self):
        try:
            values, _ = next(self.values_iter)
        except StopIteration:
            return self.last_values
        else:
            self.last_values = values
            return values

    def get_policy(self):
        policy = []
        for s in range(self.gw.n_states):
            next_states = np.array([self.gw._next_state(a,s) for a in
                                   range(self.gw.n_actions)])
            q_s = np.take(self.last_values, next_states)
            best_actions = np.argwhere(q_s == np.max(q_s)).reshape(-1)
            policy.append(best_actions)
        return policy

    def get_rewards(self):
        return self.gw.R

    def __init__(self, gw):
        super().__init__(gw)
        self.values_iter = gw.mdp.value_iteration()
        self.last_values = None

    def get_values(self):
        try:
            values, _ = next(self.values_iter)
        except StopIteration:
            return self.last_values
        else:
            self.last_values = values
            return values

    def get_policy(self):
        policy = []
        for s in range(self.gw.n_states):
            next_states = np.array([self.gw._next_state(a,s) for a in
                                   range(self.gw.n_actions)])
            q_s = np.take(self.last_values, next_states)
            best_actions = np.argwhere(q_s == np.max(q_s)).reshape(-1)
            policy.append(best_actions)
        return policy

    def get_rewards(self):
        return self.gw.R

class PolicyIterationViewer(GridWorldViewer):
    def __init__(self, gw):
        super().__init__(gw)
        self.policy_iter = gw.mdp.policy_iteration()
        self.last_values = None
        self.last_policy = None

    def get_values(self):
        try:
            values, policy, _ = next(self.policy_iter)
        except StopIteration:
            return self.last_values
        else:
            self.last_policy = policy
            self.last_values = values
            return values

    def _transform_best_actions(self, policy):
        # transform the probabilities of the policy yielded from the
        # policy iteration call into a list of best action
        transformed_policy = [
            np.argwhere(policy[s] == np.max(policy[s])).reshape(-1)
            for s in range(self.gw.n_states)
        ]
        return transformed_policy

    def get_policy(self):
        try:
            values, policy, _ = next(self.policy_iter)
        except StopIteration:
            return self.last_policy
        else:
            policy = self._transform_best_actions(policy)
            self.last_policy = policy
            self.last_values = values
            return policy

    def get_rewards(self):
        return self.gw.R

