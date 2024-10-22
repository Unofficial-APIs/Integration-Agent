from platform import node
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Set, Optional, Any
from integration_agent.util.LLM import llm


def print_dag(
    graph: nx.DiGraph,
    current_node_id: str,
    prefix: str = "",
    is_last: bool = True,
    visited: Optional[Set[str]] = None,
    depth: int = 0,
    max_depth: Optional[int] = None,
) -> None:
    """
    Recursively prints the DAG structure with visual connectors and cUrl.
    """
    if visited is None:
        visited = set()

    connector = "└── " if is_last else "├── "
    new_prefix = prefix + ("    " if is_last else "│   ")

    node_attrs = graph.nodes[current_node_id]
    dynamic_parts = node_attrs.get("dynamic_parts", [])
    key = node_attrs.get("content", "").get("key", "")
    extracted_parts = node_attrs.get("extracted_parts", [])
    input_variables = node_attrs.get("input_variables", [])
    node_type = node_attrs.get("node_type", "")  # Get node type
    
    node_label = f"[{node_type}] [node_id: {current_node_id}]"
    if input_variables:
        node_label += f"\n{new_prefix}    [input_variables: {input_variables}]"
    node_label += f"\n{new_prefix}    [dynamic_parts: {dynamic_parts}]"
    node_label += f"\n{new_prefix}    [extracted_parts: {extracted_parts}]"
    node_label += f"\n{new_prefix}    [{key}]"

    print(f"{prefix}{connector}{node_label}")

    visited.add(current_node_id)

    if max_depth is not None and depth >= max_depth:
        return

    children = list(graph.successors(current_node_id))
    child_count = len(children)

    for i, child_id in enumerate(children):
        is_last_child = i == child_count - 1

        if child_id in visited:
            loop_connector = "└── " if is_last_child else "├── "
            print(f"{new_prefix}{loop_connector}(Already visited) [node_id: {child_id}]")
        else:
            print_dag(
                graph,
                child_id,
                prefix=new_prefix,
                is_last=is_last_child,
                visited=visited,
                depth=depth + 1,
                max_depth=max_depth,
            )


def visualize_dag(graph: nx.DiGraph) -> None:
    """
    Visualizes the DAG using Matplotlib with arrows indicating direction.
    """
    plt.switch_backend("Agg")

    pos = nx.spring_layout(graph) 

    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="lightblue")

    nx.draw_networkx_edges(
        graph, pos, edgelist=graph.edges, arrowstyle="->", arrowsize=20
    )

    labels = {node: f"{node}" for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10)

    edge_labels = nx.get_edge_attributes(graph, "cUrl") 
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title("Directed Acyclic Graph (DAG)")
    plt.savefig("dag_visualization.png")
    plt.close()




def generate_code(node_id: str, graph: nx.DiGraph) -> str:
    """
    Generates Python code for a given node in the graph based on its attributes.
    """
    node_attrs = graph.nodes[node_id]
    content = node_attrs.get("content", "")
    curl = content.get("key", "")
    response = content.get("value", "")
    dynamic_parts = node_attrs.get("dynamic_parts", "")
    extracted_parts = node_attrs.get("extracted_parts", "")
    input_variables = node_attrs.get("input_variables", "")
    to_parse_response = True

    # to check if the response roughly is longer than context window
    if len(response) > 800000:
        to_parse_response = False
    
    parse_response_prompt = f"""
    The response is below:
    {response}
    
    The below variables should be parsed out of the response:
    {extracted_parts}
    """
    
    get_input_variables_prompt = f"""
    Assume these variables below are provided by the user:
    {input_variables}

    The key should be the variable name and the value should be the value to be passed in. 
    """

    prompt = f"""
    Task:
    Write me a python function with a descriptive name that makes a request like the cURL below:
    {curl}
    
    Do not hard code cookie headers. Assume cookies are in a variable called "cookie_string"

    The below variables should be passed into the request instead of hard coded:
    {dynamic_parts}
    
    {parse_response_prompt if to_parse_response else "# parse out the variables {extracted_parts} from the response"}
    
    {get_input_variables_prompt if input_variables else ""}

    Important:
    - Only output the python script and nothing else.
    - Don't include any backticks.
    - Do not use HTTP/2 pseudo-headers (those starting with ':') in the headers dictionary.
    - Remove any 'priority' headers.
    - Use standard HTTP/1.1 headers only.
    """
    
    response = llm.get_instance().invoke(prompt)
    model_output = response.content.strip()
    
    return model_output


def print_dag_in_reverse(graph: nx.DiGraph, max_depth: Optional[int] = None, to_generate_code: bool = False) -> None:
    """
    Generates the order of requests to be made based on the DAG.
    Prints the DAG starting from source nodes and ending at sink nodes, traversing successors.
    """
    generated_code = ""

    def _print_dag_recursive(
        current_node_id: str,
        prefix: str = "",
        is_last: bool = True,
        visited: Optional[Set[str]] = None,
        fully_processed: Optional[Set[str]] = None,
        depth: int = 0,
    ) -> None:
        """
        Helper function to recursively print the DAG in reverse order.
        """
        nonlocal generated_code
        if visited is None:
            visited = set()
        if fully_processed is None:
            fully_processed = set()
    
        if current_node_id in fully_processed:
            return
    
        if current_node_id in visited:
            # Avoid infinite recursion in case of cycles
            return
    
        visited.add(current_node_id)
    
        if max_depth is not None and depth >= max_depth:
            visited.remove(current_node_id)
            return
    
        # Get child nodes (successors)
        children = list(graph.successors(current_node_id))
        child_count = len(children)
    
        # Recursively process child nodes first
        for i, child_id in enumerate(children):
            is_last_child = i == child_count - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            _print_dag_recursive(
                child_id,
                prefix=new_prefix,
                is_last=is_last_child,
                visited=visited,
                fully_processed=fully_processed,
                depth=depth + 1,
            )
    
        # After all children have been processed, print the current node
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{get_node_label(graph, current_node_id)}")
        if to_generate_code:
            generated_code += generate_code(current_node_id, graph) + "\n\n"
        fully_processed.add(current_node_id)
        visited.remove(current_node_id)
    
    def get_node_label(graph: nx.DiGraph, node_id: str) -> str:
        """
        Generates a label for a node in the graph based on its attributes.
        """
        # Get node attributes
        node_attrs = graph.nodes[node_id]
        dynamic_parts = node_attrs.get("dynamic_parts", "")
        extracted_parts = node_attrs.get("extracted_parts", "")
        content = node_attrs.get("content", "")
        key = content.get("key", "")
        input_variables = node_attrs.get("input_variables", "")
        node_type = node_attrs.get("node_type", "")
        node_label = f"[{node_type}] "
        node_label += f"[node_id: {node_id}]"
        node_label += f" [dynamic_parts: {dynamic_parts}]"
        node_label += f" [extracted_parts: {extracted_parts}]"
        node_label += f" [input_variables: {input_variables}]"
        node_label += f" [{key}]"
        return node_label
    
    # Start from source nodes (nodes with no incoming edges)
    source_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    
    fully_processed = set()
    for idx, source_node in enumerate(source_nodes):
        is_last_source = idx == len(source_nodes) - 1
        _print_dag_recursive(
            source_node,
            prefix="",
            is_last=is_last_source,
            visited=set(),
            fully_processed=fully_processed,
            depth=0,
        )
    
    if to_generate_code:
        with open("generated_code.txt", "w") as f:
            f.write(generated_code)
        print("Generated code has been saved to 'generated_code.txt'")
