import requests
import json
import tiktoken
import weaviate
import os
import openai

from github import Github
from github import Auth
from github.ContentFile import ContentFile
from weaviate.util import generate_uuid5

from typing import Type, Optional
from pydantic import BaseModel, Field
from superagi.tools.base_tool import BaseTool

class RicaiCodestoreClassicSearchSchema(BaseModel):
    repository: str = Field(..., description="The Github repository to use for testing")
    location: str = Field(..., description="The path to file(s) to be found")

class RicaiCodestoreClassicSearchTool(BaseTool):
    """
    RicAI Search code by location tool

    Attributes:
        name : The name.
        description : The description.
        args_schema : The args schema.
    """
    name = "RicaiCodestoreClassicSearch"
    description = (
        "A tool for searching the code database by specific location."
    )
    args_schema: Type[RicaiCodestoreClassicSearchSchema] = RicaiCodestoreClassicSearchSchema

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self):
        """
        Initializes the RicaiCodestoreHelper with the provided Weaviate and Github credentials.

        Args:
            Weaviate_url (str): weaviate database connection url.
            Weaviate_api_key (str): weaviate database connection API key.
            Openai_api_key (str): OpenAI API key.
        """
        self.w_client = weaviate.Client(
            url=self.get_tool_config("WEAVIATE_URL"),
            auth_client_secret=weaviate.AuthApiKey(
                api_key=self.get_tool_config("WEAVIATE_API_KEY")
            ),
            additional_headers={
                "X-OpenAI-Api-Key": self.get_tool_config("OPENAI_API_KEY")
            },
        )

        self.openai_key = self.get_tool_config("OPENAI_API_KEY")

        class_obj = {
            "class": "Codefile",
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "gpt-3.5-turbo-16k",
                },
            },
            "properties": [
                {
                    "name": "file_path",
                    "dataType": ["text"],
                    "description": "Path to code file",
                },
                {
                    "name": "github_url",
                    "dataType": ["text"],
                    "description": "Github url of code file",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True
                        }
                    },
                },
                {
                    "name": "type",
                    "dataType": ["text"],
                    "description": "Type of the file",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True
                        }
                    },
                },
                {
                    "name": "repo",
                    "dataType": ["text"],
                    "description": "The code repository in Github",
                    "moduleConfig": {
                        "text2vec-openai": {
                            "skip": True
                        }
                    },
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "File content (code)",
                },
            ],
        }
        self.w_client.schema.create_class(class_obj)

        # Github setup
        ghub_auth = Auth.Token(self.get_tool_config("GITHUB_ACCESS_TOKEN"))
        ghub = Github(auth=ghub_auth)
        self.ghub_user = ghub.get_user(self.get_tool_config("GITHUB_USERNAME"))

    def _execute(self, repository: str, location: str):
        """
        Execute the RicAI Search code by location tool.

        Args:
            repository: The Github repository to use for testing
            location: The path to file(s) to be found

        Returns:
            List of files if successful. or error message.
        """

        def retrieve_code_by_location(self, ghub_repo_name, location):
            where_filter = {
                "operator": "And",
                "operands": [
                    {
                        "path": ["repo"],
                        "operator": "Equal",
                        "valueText": ghub_repo_name
                    },
                    {
                        "path": ["file_path"],
                        "operator": "Like",
                        "valueText": f"*{location}*"
                    }
                ]
            }
            response = (
                self.w_client.query
                    .get("Codefile", ["file_path", "github_url", "type", "repo", "content"])
                    .with_where(where_filter)
                    .do()
            )
            files = response["data"]["Get"]["Codefile"]
            return json.dumps(files)

        try:
            result = retrieve_code_by_location(repository, location)
            return result
        except Exception as err:
            return f"Error: Unable to search codebase - {err}"