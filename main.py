from sys import argv
from gridworld import GridWorld
from viewer import *

class GridWorldError(Exception): pass

gw = None

def print_help():
    print('''
commands:
? - this help
w <width> - set width
h <height> - set height
t [<state>] - if state is specified, toggle terminal state; otherwise,
print all terminal states
r <state> <reward> [<action>] - set reward for taking action in state
(reward will be set for all actions if none is specified)
s <state> <next_states> [<probabilities>] - set transition probabilities
for state
(if probabilities not specified they will be equally likely)
eval - run policy evaluation (for random policy)
pi - run policy iteration
vi - run value iteration
''')

def get_arg(args, n, type, element_type=None):
    if type is list:
        vals = args[n].split(',')
        try:
            return [element_type(v) for v in vals]
        except ValueError:
            raise GridWorldError('invalid arguments')
        
    try:
        return type(args[n])
    except (IndexError, ValueError):
        raise GridWorldError('invalid arguments')

def get_optional_arg(args, n, type, element_type=None, default=None):
    if len(args)-1 < n:
        return default

    if type is list:
        vals = args[n].split(',')
        try:
            return [element_type(v) for v in vals]
        except ValueError:
            raise GridWorldError('invalid arguments')

    try:
        return type(args[n])
    except ValueError:
        raise GridWorldError('invalid arguments')

def process_line(line):
    global gw
    tokens = line.split(' ')
    cmd, *args = tokens

    if cmd == '?':
        print_help()

    elif cmd == 'w':
        w = get_optional_arg(args, 0, int)
        if w is None:
            print(gw.w)
        else:
            gw.set_w(w)

    elif cmd == 'h':
        h = get_optional_arg(args, 0, int)
        if h is None:
            print(gw.h)
        else:
            gw.set_h(h)

    elif cmd == 't':
        states = get_optional_arg(args, 0, list, int)
        if states == None:
            print(gw.terminal_states)
        else:
            for s in states:
                if s < 0 or s >= gw.n_states:
                    raise GridWorldError('invalid state number')

                try:
                    gw.toggle_terminal_state(s)
                except ValueError as e:
                    raise GridWorldError(e)

    elif cmd == 'r':
        states = get_arg(args, 0, list, int)
        reward = get_arg(args, 1, int)
        actions = get_optional_arg(args, 2, list, int, default=(0,1,2,3))

        for s in states:
            for a in actions:
                gw.set_reward(s, reward, a)

    elif cmd == 's':
        # TODO
        pass

    elif cmd == 'vi':
        ValueIterationViewer(gw).run()

    elif cmd == 'pi':
        PolicyIterationViewer(gw).run()

    elif cmd == 'q':
        exit()

    else:
        raise GridWorldError('unknown command')

if __name__ == '__main__':
    try:
        w, h = int(argv[1]), int(argv[2])
    except (ValueError, IndexError):
        w = h = 4
    gw = GridWorld(w, h, [0])

    while True:
        line = input('> ')
        try:
            process_line(line.lstrip().rstrip())
        except GridWorldError as e:
            print(e)
            continue
