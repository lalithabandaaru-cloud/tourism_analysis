from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

repo_id="LalithaRB/tourism-analysis"
repo_type="dataset"

# Initialize api client
api=HfApi(token=os.getenv("HF_TOKEN"))

#step1 : check if the space exists
try:
  api.repo_info(repo_id=repo_id, repo_type=repo_type)
  print(f"Space '{repo_id}' already exists Using it.")
except RepositoryNotFoundError:
  print(f"Space '{repo_id}' not found. Creating new space..")
  create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
  print(f"Space '{repo_id}' created successfully")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    ignore_patterns=".gitattributes",
)
