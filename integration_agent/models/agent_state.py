from typing import List, Optional, TypedDict, Dict

class AgentState(TypedDict):
    master_node: str  # User request
    in_process_node: str
    to_be_processed_nodes: List[str]
    in_process_node_dynamic_parts: List[str]
    action_url: str
    input_variables: Dict[str, str]
