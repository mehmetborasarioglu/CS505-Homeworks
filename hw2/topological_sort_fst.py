# SYSTEM IMPORTS


# PYTHON PROJECT IMPORTS


class SearchColors(object):
    WHITE = 0
    GRAY = 1
    BLACK = 2


class TopologicalPacket(object):
    def __init__(self):
        self.color = SearchColors.WHITE
        self.discovery_time = 0
        self.finishing_time = 0
        self.parent = None


def fst_dfs_visit(fst, state_dfs_map, order, state, time):
    time += 1

    # discover the node
    state_dfs_map[state].discovery_time = time
    state_dfs_map[state].color = SearchColors.GRAY
    for transition, _ in fst.transitions_from[state].items():
        # visit all unvisited children
        if state_dfs_map[transition.r].color == SearchColors.WHITE:
            state_dfs_map[transition.r].parent = state
            fst_dfs_visit(fst, state_dfs_map, order, transition.r, time)
    # finish the node
    state_dfs_map[state].color = SearchColors.BLACK
    time += 1
    state_dfs_map[state].finishing_time = time
    order.insert(0, state)


def fst_dfs_search(fst, order):
    time = 0

    # make sure we unvisit all nodes
    state_dfs_map = {name: TopologicalPacket() for name in fst.states}

    # perform the depth first search
    # for state in fst.states:
    #     if state_dfs_map[state].color == SearchColors.WHITE:
    #         fst_dfs_visit(fst, state_dfs_map, order, state, time)
    fst_dfs_visit(fst, state_dfs_map, order, fst.start, time)

def fst_topological_sort(fst):
    topological_order = list()
    fst_dfs_search(fst, topological_order)
    return topological_order    

