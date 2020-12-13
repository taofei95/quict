from .._mcts_base import *

class TreeBasedMCTS(MCTSBase):
    def __init__(self, play_times: int =20, mode: str = None, mode_sim :List[int] = None,
                 coupling_graph : List[Tuple] = None, **params):
        super.__init__(coupling_graph)
        self.play_times = play_times 
        self.mode = mode
        self.mode_sim = mode_sim
    
    def search(self, root_node : MCTSNode)):
        pass


    def _expand(self):
        """
        open all child nodes of the  current node by applying the swap gates in candidate swap list
        """
        pass

    def _rollout(self, method: str):
        """
        complete a heuristic search from the current node
        """
        pass
    
    def _backpropagate(self, reward: float):
        """
        use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        pass

    def _select(self):
        """
        select the child node with highest score
        """
        pass

    def _eval(self):
        """
        evaluate the vlaue of the current node by DNN method
        """
        pass
    
    
    


        


    


