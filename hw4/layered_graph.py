# SYSTEM IMPORTS
from collections.abc import Mapping, Sequence
from collections import defaultdict
from typing import Tuple, Type


# PYTHON PROJECT IMPORTS


LayeredGraphType = Type["LayeredGraph"]

"""
    A LayeredGraph object is used to build the graph-expansion that we see in the lecture slides when using a HMM
    for decoding/forward/backward traversals. I have implemented a LayeredGraph as a list of dictionaries. Each
    dictionary is a layer, where the dictionary at layer-index 0 is the first layer in the graph (typically containing
    a single vertex for <BOS>), the dictionary at layer-index 1 is the layer suceeding <BOS>, etc. The last layer
    is a dictionary that should only have a single vertex in it for <EOS>.

    Each dictionary maps vertex names to a pair of values. This pair is used to store two values:
        - the "value" of the vertex. This value is whatever value your specific traversal is trying to calculate.
          For instance, if we're using a LayeredGraph to do viterbi decoding, then the "value" of a vertex
          would be the weight of the largest weighted path from <BOS> to that vertex. If we were instead doing
          a HMM forward traversal, then the "value" of a vertex would be the sum of all path weights from <BOS>
          to that vertex. This "value" is just a placeholder for what you're trying to calculate per vertex.

        - The "parent" of the vertex. This is for backpointers. This value in the pair stores the name of the vertex
          in the previous layer who is the backpointer for the current vertex.

    This class does not preallocate the dictionaries for you, so if you want to add a layer to a LayeredGraph
    instance, you will have to call <instance>.add_layer()
"""
class LayeredGraph(object):

    NODE_VALUE_TUPLE_INDEX: int = 0
    NODE_PARENT_TUPLE_INDEX: int = 1

    def __init__(self: LayeredGraphType,
                 init_val: float = 0.0          # the initial value of a vertex
                 ) -> None:
        self.node_layers: Sequence[Mapping[str, Tuple[float, str]]] = list()
        self.init_val: float = init_val

    def add_layer(self: LayeredGraphType) -> None:
        self.node_layers.append(defaultdict(lambda: tuple([self.init_val, None])))

    """
        This method will add a new vertex to the *last* layer of the graph with the specified vertex name,
        vertex "value", and backpointer.
    """
    def add_node(self: LayeredGraphType,
                 child_node_name: str,
                 child_value: float,
                 parent_node_name: str
                 ) -> None:
        current_layer: Mapping[str, Tuple[float, str]] = self.node_layers[-1]
        current_layer[child_node_name] = tuple([child_value, parent_node_name])

    def get_node_in_layer(self: LayeredGraphType,
                          layer_idx: int,
                          node_name: str
                          ) -> Tuple[float, str]:
        return self.node_layers[layer_idx][node_name]

    def get_node_in_last_layer(self: LayeredGraphType,
                               node_name: str
                               ) -> Tuple[float, str]:
        return self.get_node_in_layer(-1, node_name)

