# Integration Agent

An AI agent that reverse-engineers third-party platform's internal APIs.

## What Integration Agent Does

When provided with a .har file containing all browser network requests and a prompt describing the desired action, the agent outputs the dependency graph of all network requests. It can also generate code that hits all the endpoints, though the code generation feature is still experimental.

The goal of this agent is to assist in creating integrations via reverse engineering APIs. We're working to fully automate the creation of integration code.

## How It Works

The agent mimics what a human does when reverse-engineering. Here’s how the agent works, using an example. Let's assume we want to download utility bills from a utility website:

1. You provide a .har file with all network requests and a prompt about your desired action(s) — in this case, to download utility bills.
2. The agent identifies the final request that downloads utility bills. 
   For example, the request URL might look like this:
   ```
   https://www.example.com/utility-bills?accountId=123&userId=456
   ```
   
3. It then identifies parts of the request that depend on other requests. The example URL contains dynamic parts that must be obtained from other requests.
   ```
   accountId=123 userId=456
   ```
4. It finds other requests whose response contains any of these 2 dynamic parts.
For example, the newly found request URLs might look like this:
   ```
   GET https://www.example.com/getAccountId
   GET https://www.example.com/getUserId
   ```
5. This process repeats until the most recently found request doesn’t depend on any other request.
6. The agent outputs a graph of the network request dependencies, which you can use to create your integration.

## Features

- Generate a dependency graph of requests to make the final request that performs the desired action.
- Allow input variables (for example, providing the YEAR to download a document from).
- Generate code to hit all requests in the graph to perform the final action (still experimental).

## Demo

[![Integration Agent Demo](https://img.youtube.com/vi/tP8BrpiSUSY/0.jpg)](https://www.youtube.com/watch?v=tP8BrpiSUSY)

## Setup

1. Set up your OpenAI [API Keys](https://platform.openai.com/account/api-keys) and add the `OPENAI_API_KEY` environment variable.
2. Install Python requirements via poetry:
   ```
   poetry install
   ```
3. Open a poetry shell:
   ```
   poetry shell
   ```
4. Run the following command to spawn a browser:
   ```
   poetry run python create_har.py
   ```
   Log into your platform and perform the desired action (such as downloading a utility bill).
5. Run the Integration Agent:
   ```
   poetry run python -m integration_agent --prompt "download utility bills"
   ```
    You can also run it via Jupyter Notebook `main.ipynb`

## Usage

After setting up the project, you can use the Integration Agent to analyze and reverse-engineer API requests for various platforms. Simply provide the appropriate .har file and a prompt describing the desired action.

```
poetry run python -m integration_agent --help
Usage: python -m integration_agent [OPTIONS]

Options:
  --model TEXT                    The LLM model to use
  --prompt TEXT                   The prompt for the model  [required]
  --har-path TEXT                 The HAR file path
  --cookie-path TEXT              The cookie file path
  --max_steps INTEGER             The optional max_steps (default is 10)
  --input_variables <TEXT TEXT>...
                                  Input variables in the format key value
  --help                          Show this message and exit
```

## Contributing

Contributions to improve the Integration Agent are welcome. Please feel free to submit issues or pull requests on the project's repository.

## Info

Integration Agent is built by UnofficialAPIs.com. Besides our work on the agent, we offer services to build custom integrations upon request, host, and manage authentication. If you have requests or want to work with us, reach out at richard@taiki.online.


This repo is intended to be used as a package in a larger project. https://github.com/Unofficial-APIs/Integrations
