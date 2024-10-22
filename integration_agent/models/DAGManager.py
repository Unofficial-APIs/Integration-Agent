from typing import List, Optional, Literal, Dict # Import Literal for type enforcement
import networkx as nx
import uuid


class DAGManager:
    NODE_TYPES = {"cookie", "master", "cURL"}  
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.root_id = None 
    def add_node(
        self,
        node_type: Literal["cookie", "master", "cURL", "not found"],  
        content: Optional[dict] = None,  
        dynamic_parts: Optional[List[str]] = None,
        extracted_parts: Optional[List[str]] = None,
        input_variables: Optional[Dict[str, str]] = None,
    ):
        node_id = str(uuid.uuid4())
        self.graph.add_node(node_id, node_type=node_type, content=content, dynamic_parts=dynamic_parts, extracted_parts=extracted_parts, input_variables=input_variables)
        return node_id
    
    def update_node(
        self, 
        node_id: str, 
        **attributes: Optional[List[str]]):
        
        for attr, value in attributes.items():
            if value is not None:
                self.graph.nodes[node_id][attr] = value

    def detect_cycles(self):
        """
        Detects if there are cycles in the DAG managed by this class.
        If a cycle is found, it returns the list of nodes involved in the cycle.
        If no cycle is found, it returns None.

        Returns:
        - A list of nodes forming a cycle, or None if no cycles are found.
        """
        try:
            cycle = list(nx.find_cycle(self.graph, orientation='original'))
            print("Cycle detected:")
            return cycle
        except nx.exception.NetworkXNoCycle:
            print("No cycles found in the graph.")
            return None
        
    def get_node(self, node_id: str) -> Optional[Dict]:
        """
        Retrieves the attributes of the specified node.
        
        :param node_id: ID of the node to retrieve.
        :return: Dictionary of node attributes or None if the node does not exist.
        """
        return self.graph.nodes.get(node_id, None)
    
    def add_edge(self, from_node_id: str, to_node_id: str):
        self.graph.add_edge(from_node_id, to_node_id)

    def __str__(self):
        nodes_info = []
        for node_id in self.graph.nodes:
            attrs = self.graph.nodes[node_id]
            nodes_info.append(f"{node_id}: {attrs}")
        return "\n".join(nodes_info)


