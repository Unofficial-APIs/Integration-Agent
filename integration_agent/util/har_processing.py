import json
import os
from urllib.parse import urlparse
from integration_agent.models.request import Request
from typing import Tuple, Dict, Optional, Any, List

excluded_keywords = (
    "google",
    "taboola",
    "datadog",
    "sentry",
)

def format_request(har_request: Dict[str, Any]) -> Request:
    """
    Formats a HAR request into a Request object.
    """
    method = har_request.get("method", "GET")
    url = har_request.get("url", "")

    # Store headers as a dictionary
    headers = {
        header.get("name", ""): header.get("value", "")
        for header in har_request.get("headers", [])
    }

    query_params_list = har_request.get("queryString", [])
    query_params = {param["name"]: param["value"] for param in query_params_list} if query_params_list else None

    post_data = har_request.get("postData", {})
    body = post_data.get("text") if post_data else None

    # Try to parse body as JSON if Content-Type is application/json
    if body:
        headers_lower = {k.lower(): v for k, v in headers.items()}
        content_type = headers_lower.get('content-type')
        if content_type and 'application/json' in content_type.lower():
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                pass  # Keep body as is if not valid JSON

    return Request(
        method=method,
        url=url,
        headers=headers, 
        query_params=query_params,
        body=body
    )


def format_response(har_response: Dict[str, Any]) -> str:
    """
    Extracts and returns the content text from a HAR response.
    """
    content = har_response.get("content", {}).get("text", "")
    return content


def parse_har_file(har_file_path: str) -> Dict[Request, str]:
    """
    Parses the HAR file and returns a dictionary mapping Request objects to response strings.
    """
    req_res_dict = {}

    with open(har_file_path, 'r', encoding='utf-8') as file:
        har_data = json.load(file)

    entries = har_data.get("log", {}).get("entries", [])

    for entry in entries:
        request_data = entry.get("request", {})
        response_data = entry.get("response", {})

        formatted_request = format_request(request_data)
        response_string = format_response(response_data)

        req_res_dict[formatted_request] = response_string

    return req_res_dict


def build_url_to_req_res_map(req_res_dict: Dict[Request, str]) -> Dict[str, Dict[str, Any]]:
    """
    Builds a dictionary mapping URLs to {'request': formatted_request, 'response': response_string}
    """
    url_to_req_res_dict = {}

    for request, response in req_res_dict.items():
        url = request.url
        # If multiple requests to the same URL, you can choose to overwrite or store all
        url_to_req_res_dict[url] = {
            'request': request,
            'response': response
        }

    return url_to_req_res_dict


def get_har_urls(har_file_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Extracts and returns a list of tuples containing method, URL, response format, and response preview
    from a HAR file, excluding certain file types and keywords.
    """
    # List to store tuples of URLs, request methods, response file formats, and response preview
    urls_with_details = []

    # Define a tuple of file extensions to exclude
    excluded_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".svg",
        ".ico",  # Image files
        ".css",  # Stylesheets
        # ".js",
        # ".map",  # JavaScript files
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",  # Font files
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".flv",
        ".wmv",
        ".webm",  # Media files
        # ".pdf",
        # ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".exe",
        ".dmg",  # Other non-text files
    )

    # Read the HAR file
    with open(har_file_path, "r", encoding="utf-8") as file:
        har_data = json.load(file)

    # Extract entries from the HAR data
    entries = har_data.get("log", {}).get("entries", [])   
    for entry in entries:
        request = entry.get("request", {})
        response = entry.get("response", {})
        url = request.get("url")
        method = request.get("method", "GET")  # Default to 'GET' if method is missing
        response_format = response.get("content", {}).get("mimeType", "")
        response_text = response.get("content", {}).get("text", "")
        response_preview = response_text[:30] if response_text else ""

        if url:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()

            _, extension = os.path.splitext(path)

            request_text = url.lower()

            headers = request.get("headers", [])
            for header in headers:
                request_text += header.get("name", "").lower()
                request_text += header.get("value", "").lower()

            postData = request.get("postData", {}).get("text", "").lower()
            request_text += postData

            # Exclude URLs with the specified extensions or if keywords are in the request
            # this is done to reduce the number of requests we send to the LLM
            if extension not in excluded_extensions and not any(
                keyword.lower() in request_text for keyword in excluded_keywords
            ):
                urls_with_details.append((method, url, response_format, response_preview))

    return urls_with_details
    

def parse_cookie_file_to_dict(cookie_file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses a JSON cookie file and returns a dictionary of cookie data.
    """
    parsed_data = {}

    with open(cookie_file_path, "r") as file:
        cookies = json.load(file)  

    for cookie in cookies:
        name = cookie.get("name")
        value = cookie.get("value")
        domain = cookie.get("domain")
        path = cookie.get("path")

        if name:
            parsed_data[name] = {
                "value": value,
                "domain": domain,
                "path": path,
                "expires": cookie.get("expires"),
                "httpOnly": cookie.get("httpOnly"),
                "secure": cookie.get("secure"),
                "sameSite": cookie.get("sameSite"),
            }

    return parsed_data
