from typing import List, Dict, Optional, Any
import json

class Request:
    def __init__(self, method: str, url: str, headers: Dict[str, str], 
                 query_params: Optional[Dict[str, str]] = None, body: Optional[Any] = None):
        self.method = method
        self.url = url
        self.headers = headers  
        self.query_params = query_params
        self.body = body

    def to_curl_command(self) -> str:
        curl_parts = [f"curl -X {self.method}"]

        for name, value in self.headers.items():
            curl_parts.append(f"-H '{name}: {value}'")

        if self.query_params:
            query_string = "&".join([f"{k}={v}" for k, v in self.query_params.items()])
            self.url += f"?{query_string}"

        if self.body:
            content_type = None
            for k in self.headers:
                if k.lower() == 'content-type':
                    content_type = self.headers[k]
                    break

            if isinstance(self.body, dict):
                # Add Content-Type header if not present
                if not content_type:
                    curl_parts.append(f"-H 'Content-Type: application/json'")
                curl_parts.append(f"--data '{json.dumps(self.body)}'")
            elif isinstance(self.body, str):
                curl_parts.append(f"--data '{self.body}'")

        curl_parts.append(f"'{self.url}'")

        return " ".join(curl_parts)

    def to_minified_curl_command(self) -> str:
        """
        Minifies the curl command by removing referer and cookie headers.
        This is done to reduce LLM hallucinations.
        """
        curl_parts = [f"curl -X {self.method}"]

        for name, value in self.headers.items():
            if name.lower() not in ['referer', 'cookie']:
                curl_parts.append(f"-H '{name}: {value}'")

        if self.query_params:
            query_string = "&".join([f"{k}={v}" for k, v in self.query_params.items()])
            self.url += f"?{query_string}"

        if self.body:
            content_type = None
            for k in self.headers:
                if k.lower() == 'content-type':
                    content_type = self.headers[k]
                    break

            if isinstance(self.body, dict):
                if not content_type:
                    curl_parts.append(f"-H 'Content-Type: application/json'")
                curl_parts.append(f"--data '{json.dumps(self.body)}'")
            elif isinstance(self.body, str):
                curl_parts.append(f"--data '{self.body}'")

        curl_parts.append(f"'{self.url}'")

        return " ".join(curl_parts)

    def __str__(self) -> str:
        return self.to_curl_command()
