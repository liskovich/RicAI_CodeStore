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

class RicaiCodestoreUpdateSchema(BaseModel):
    repository: str = Field(..., description="The Github repository to use for testing")

class RicaiCodestoreUpdateTool(BaseTool):
    """
    RicAI Update Database with latest code tool

    Attributes:
        name : The name.
        description : The description.
        args_schema : The args schema.
    """
    name = "RicaiCodestoreUpdate"
    description = (
        "A tool for updating the code database with latest code from Github."
    )
    args_schema: Type[RicaiCodestoreUpdateSchema] = RicaiCodestoreUpdateSchema

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

    def _execute(self, repository: str):
        """
        Execute the RicAI Update Database with latest code tool.

        Args:
            repository: The Github repository to use for testing

        Returns:
            Nothing if successful. or error message.
        """
        def upsert_codebase(ghub_repo_name):
            repo = self.ghub_user.get_repo(ghub_repo_name)
            contents = repo.get_contents("")
            codefiles = []

            # TODO: populate with more ignorable file types
            ignore_extensions = [
                ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico", ".svg",
                ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv",
                ".mp3", ".wav", ".ogg", ".aac", ".flac", ".wma",
                ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".odt", ".ods", ".odp",
                ".zip", ".rar", ".tar", ".gz", ".7z",
                ".exe", ".dll", ".app", ".apk", ".iso", ".img", ".dmg",
            ]

            while contents:
                file = contents.pop(0)
                if file.type == "dir":
                    contents.extend(repo.get_contents(file.path))
                else:
                    extension = os.path.splitext(file.name)[1].lower()
                    if extension not in ignore_extensions:
                        print(file.path)
                        file_content = file.decoded_content.decode("utf-8")
                        codefiles.append({
                            "file_path": file.path,
                            "github_url": file.url,
                            "type": file.type,
                            "repo": file.repository,
                            "content": file_content
                        })
                        file.last_modified

            # TODO: check if code from specific repo/codebase in Github is already present in the vector database
            # TODO: make sure that deterministic uuid generation works 
            class_name = "Codefile"
            with self.w_client.batch() as batch:
                for codefile in codefiles:
                    batch.add_data_object(
                        codefile, 
                        class_name,
                        uuid=generate_uuid5(identifier=codefile["path"], namespace=codefile["repo"])
                    )
            return True
    
        try:                        
            result = upsert_codebase(repository)
            return f'Codebase updated successfully - {result}'
        except Exception as err:
            return f"Error: Unable to update codebase with latest code - {err}"