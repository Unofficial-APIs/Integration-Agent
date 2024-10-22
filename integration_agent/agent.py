import json
import urllib
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

from integration_agent.util.LLM import llm
from integration_agent.models.DAGManager import DAGManager
from integration_agent.util.har_processing import *
from integration_agent.models.request import Request
from integration_agent.models.agent_state import AgentState

class IntegrationAgent:
    ACTION_URL_KEY: str = "action_url"
    IN_PROCESS_NODE_KEY: str = "in_process_node"
    TO_BE_PROCESSED_NODES_KEY: str = "to_be_processed_nodes"
    IN_PROCESS_NODE_DYNAMIC_PARTS_KEY: str = "in_process_node_dynamic_parts"
    MASTER_NODE_KEY: str = "master_node"
    INPUT_VARIABLES_KEY: str = "input_variables"

    def __init__(
        self,
        prompt: str,
        har_file_path: str,
        cookie_path: str,
    ):  
        self.prompt: str = prompt
        self.duplicate_part_set: Set[str] = set()
        self.global_master_node: Optional[str] = None
        self.req_to_res_map: Dict[Request, str] = parse_har_file(har_file_path)
        self.url_to_res_req_dict: Dict[str, Dict[str, Any]] = build_url_to_req_res_map(self.req_to_res_map)
        self.har_urls: List[Tuple[str, str, str, str]] = get_har_urls(har_file_path)
        self.cookie_dict: Dict[str, Dict[str, Any]] = parse_cookie_file_to_dict(cookie_path)
        self.curl_to_id_dict: Dict[str, str] = {}
        self.cookie_to_id_dict: Dict[str, str] = {}
        self.dag_manager: DAGManager = DAGManager()

    def end_url_identify_agent(self, state: AgentState) -> AgentState:
        """
        Identify the URL responsible for a specific action
        """
        function_def = {
            "name": "identify_end_url",
            "description": "Identify the URL responsible for a specific action",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": f"The URL responsible for {self.prompt}"
                    }
                },
                "required": ["url"]
            }
        }

        prompt = f"""
        {self.har_urls}
        Task:
        Given the above list of URLs, request types, and response formats, find the URL responsible for the action below:
        {self.prompt}
        """

        response = llm.get_instance().invoke(
            prompt,
            functions=[function_def],
            function_call={"name": "identify_end_url"}
        )
        
        function_call = response.additional_kwargs['function_call']
        end_url = json.loads(function_call['arguments'])['url']

        state[self.ACTION_URL_KEY] = end_url
        return state

    def input_variables_identifying_agent(self, state: AgentState) -> AgentState:
        """
        Identify input variables present in the cURL command
        """
        in_process_node_id = state[self.IN_PROCESS_NODE_KEY]
        curl = self.dag_manager.graph.nodes[in_process_node_id]["content"]["key"].to_curl_command()
        input_variables = state[self.INPUT_VARIABLES_KEY]
        if not input_variables:
            return state
        
        function_def = {
            "name": "identify_input_variables",
            "description": "Identify input variables present in the cURL command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "identified_variables": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "variable_name": {"type": "string", "description": "The original key of the variable"},
                                "variable_value": {"type": "string", "description": "The exact version of the variable that is present in the cURL command. This should closely match the value in the provided Input Variables."}
                            },
                            "required": ["variable_name", "variable_value"]
                        },
                        "description": "A list of identified variables and their values."
                    }
                },
                "required": ["identified_variables"]
            }
        }


        prompt = f"""
        cURL: {curl}
        Input Variables: {input_variables}

        Task:
        Identify which input variables (the value in the key-value pair) from the Input Variables provided above are present in the cURL command.

        Important:
        - If an input variable is found in the cURL, include it in the output.
        - Do not include variables that are not provided above.
        - The key of the input variable is a description of the variable.
        - The value is the value that should closely match the value in the cURL command. No substitutions.

        """


        response = llm.get_instance().invoke(
            prompt,
            functions=[function_def],
            function_call={"name": "identify_input_variables"}
        )

        function_call = response.additional_kwargs.get('function_call', {})
        arguments = json.loads(function_call.get('arguments', '{}'))
        identified_variables = arguments.get('identified_variables', [])
        
        if identified_variables:
            # Convert the identified_variables format
            converted_variables = {item['variable_name']: item['variable_value'] for item in identified_variables}
            
            current_dynamic_parts = self.dag_manager.graph.nodes[in_process_node_id].get("dynamic_parts", [])
            updated_dynamic_parts = [part for part in current_dynamic_parts if part not in converted_variables.values()]
            self.dag_manager.update_node(in_process_node_id, dynamic_parts=updated_dynamic_parts, input_variables=converted_variables)

        return state

    def dynamic_part_identifying_agent(self, state: AgentState) -> AgentState:
        """
        Identify dynamic parts present in the cURL command
        """
        in_process_node_id = state[self.TO_BE_PROCESSED_NODES_KEY].pop()
        request = self.dag_manager.graph.nodes[in_process_node_id]["content"]["key"]
        curl = request.to_minified_curl_command()
        if curl.endswith(".js'"):
            self.dag_manager.update_node(in_process_node_id, dynamic_parts=[])
            state[self.IN_PROCESS_NODE_DYNAMIC_PARTS_KEY] = [] 
            state[self.IN_PROCESS_NODE_KEY] = in_process_node_id
            return state
        

        input_variables = state[self.INPUT_VARIABLES_KEY]            

        function_def = {
            "name": "identify_dynamic_parts",
            "description": (
                "Given the above cURL command, identify which parts are dynamic and validated by the server "
                "for correctness (e.g., IDs, tokens, session variables). Exclude any parameters that represent "
                "arbitrary user input or general data that can be hardcoded (e.g., amounts, notes, messages)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dynamic_parts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of dynamic parts identified in the cURL command. Do not include duplicates. "
                            "Only include the values (not the keys or any not extra part in front and after the value) of parts that are unique to a user or session "
                            "and, if incorrect, will cause the request to fail."
                            "nothing should be part of the cookies header or the referer header"
                        ),
                    }
                },
                "required": ["dynamic_parts"],
            },
        }

        prompt = f"""
        URL: {curl}

        Task:

        Use your best judgment to identify which parts of the cURL command are dynamic, specific to a user or session, and are checked by the server for validity. These include tokens, IDs, session variables, or any other values that are unique to a user or session and, if incorrect, will cause the request to fail.

        Important:
            - IGNORE THE COOKIE HEADER
            - Ignore common headers like user-agent, sec-ch-ua, accept-encoding, referer, etc.
            - Exclude parameters that represent arbitrary user input or general data that can be hardcoded, such as amounts, notes, messages, actions, etc.
            - Only output the variable values and not the keys.
            - Only include dynamic parts that are unique identifiers, tokens, or session variables.

        """

        response = llm.get_instance().invoke(
            prompt,
            functions=[function_def],
            function_call={"name": "identify_dynamic_parts"}
        )

        function_call = response.additional_kwargs['function_call']
        dynamic_parts = json.loads(function_call['arguments'])['dynamic_parts']

        self.dag_manager.update_node(in_process_node_id, dynamic_parts=dynamic_parts)

        # to detect if input_variables are in the request
        present_variables = [variable for variable in input_variables if variable in curl]
        if present_variables:
            for variable in present_variables:
                if variable in dynamic_parts:
                    dynamic_parts.remove(variable)
            self.dag_manager.update_node(in_process_node_id, input_variables=present_variables)


        state[self.IN_PROCESS_NODE_DYNAMIC_PARTS_KEY] = dynamic_parts
        state[self.IN_PROCESS_NODE_KEY] = in_process_node_id
        return state

    def url_to_curl(self, state: AgentState) -> AgentState:
        """
        Identify the master cURL command responsible for the action
        """
        request = self.url_to_res_req_dict[state["action_url"]]["request"]
        curl = request.to_curl_command()
        if curl in self.curl_to_id_dict:
            master_node_id = self.curl_to_id_dict[curl]
        else:
            master_node_id = self.dag_manager.add_node(
                node_type="master_curl",  # Specify node type
                content={
                    "key": request,
                    "value": self.req_to_res_map[request]
                },
                dynamic_parts=["None"],
                extracted_parts=["None"]
            )
            self.curl_to_id_dict[curl] = master_node_id
        state[self.MASTER_NODE_KEY] = master_node_id
        state[self.TO_BE_PROCESSED_NODES_KEY].append(master_node_id)
        self.global_master_node_id = master_node_id
        return state

    def get_simplest_request(self, request_list: List[Request]) -> Request:
        """
        Find the index of the simplest cURL command from a list
        """
        function_def = {
            "name": "get_simplest_curl_index",
            "description": "Find the index of the simplest cURL command from a list",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The index of the simplest cURL command in the list"
                    }
                },
                "required": ["index"]
            }
        }
        # convert request objects to strings
        serializable_list = [str(req) for req in request_list]

        prompt = f"""
        {json.dumps(serializable_list)}
        Task:
        Given the above list of cURL commands, find the index of the curl that has the least number of dependencies and variables.
        The index should be 0-based (i.e., the first item has index 0).
        """

        response = llm.get_instance().invoke(
            prompt,
            functions=[function_def],
            function_call={"name": "get_simplest_curl_index"}
        )

        function_call = response.additional_kwargs['function_call']
        simplest_curl_index = json.loads(function_call['arguments'])['index']
        
        # Retrieve the actual cURL command using the index
        simplest_curl = request_list[simplest_curl_index]
        return simplest_curl

    def find_curl_from_content(self, state: AgentState) -> AgentState:
        """
        Find the cURL command that contains the dynamic parts
        """
        search_string_list = state[self.IN_PROCESS_NODE_DYNAMIC_PARTS_KEY]
        search_string_list_leftovers = search_string_list.copy()

        in_process_node_id = state[self.IN_PROCESS_NODE_KEY]
        new_to_be_processed_nodes = []

        # Handle cookies
        for search_string in search_string_list_leftovers[:]:
            cookie_key = self.find_key_by_string_in_value(
                self.cookie_dict, search_string
            )
            if cookie_key:
                search_string_list_leftovers.remove(search_string)
                if cookie_key in self.cookie_to_id_dict:
                    cookie_node_id = self.cookie_to_id_dict[cookie_key]
                else:
                    cookie_node_id = self.dag_manager.add_node(
                        node_type="cookie",  # Specify node type
                        content={
                            "key": cookie_key,
                            "value": search_string
                        }, 
                        extracted_parts=[search_string]
                    )
                    self.cookie_to_id_dict[cookie_key] = cookie_node_id
                    #dont need to add node to to_be_processed_nodes because cookies dont need further processing
                self.dag_manager.add_edge(in_process_node_id, cookie_node_id)

        # Handle curls
        if search_string_list_leftovers:
            for search_string in search_string_list_leftovers[:]:
                requests_with_search_string = []

                for request, response in self.req_to_res_map.items():
                    curl = str(request)
                    if (
                        (
                            isinstance(curl, str)
                            and search_string.lower() in response.lower()
                        )
                        and (search_string.lower() not in curl.lower())
                    ) or (
                        urllib.parse.unquote(search_string) in curl
                        and (urllib.parse.unquote(search_string) not in curl)
                    ):
                        requests_with_search_string.append(request)
                simplest_request = ""

                # Get simplest curl to reduce number of dependencies
                if len(requests_with_search_string) > 1:
                    simplest_request = self.get_simplest_request(requests_with_search_string)
                elif len(requests_with_search_string) == 1:
                    simplest_request = requests_with_search_string[0]
                else:
                    print(f"Could not find curl with search string: {search_string} in response")
                    not_found_node_id = self.dag_manager.add_node(
                        node_type="not found",
                        content={
                            "key": search_string
                        },
                    )
                    self.dag_manager.add_edge(in_process_node_id, not_found_node_id)
                    search_string_list_leftovers.remove(search_string)

                    continue
        
                        
                if simplest_request not in self.curl_to_id_dict:
                    curl_node_id = self.dag_manager.add_node(
                        node_type="curl",  # Specify node type
                        content={
                            "key": simplest_request,
                            "value": self.req_to_res_map[simplest_request]
                        },
                        extracted_parts=[search_string]
                    )
                    self.curl_to_id_dict[simplest_request] = curl_node_id
                    new_to_be_processed_nodes.append(curl_node_id)
                else:
                    # append new extracted part to existing curl node
                    curl_node_id = self.curl_to_id_dict[simplest_request]
                    node = self.dag_manager.get_node(curl_node_id)
                    new_extracted_parts = node.get("extracted_parts", [])
                    new_extracted_parts.append(search_string)
                    # Remove duplicates from new_extracted_parts
                    new_extracted_parts = list(dict.fromkeys(new_extracted_parts))
                    self.dag_manager.update_node(curl_node_id, extracted_parts=new_extracted_parts)

                self.dag_manager.add_edge(in_process_node_id, curl_node_id)
                
        state[self.TO_BE_PROCESSED_NODES_KEY].extend(new_to_be_processed_nodes)
        state[self.IN_PROCESS_NODE_DYNAMIC_PARTS_KEY] = []
        return state

    @staticmethod
    def find_key_by_string_in_value(dictionary: Dict[str, Dict[str, Any]], search_string: str) -> Optional[str]:
        for key, value in dictionary.items():
            if search_string in value.get("value", ""):
                return key
        return None
    

