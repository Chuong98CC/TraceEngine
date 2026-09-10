from huggingface_hub import HfApi

api = HfApi()

# Create the repo if it doesn't exist
REPO_ID = "Chuong98vt/TraceEngine"
FOLDER_PATH = "/home/chuong/workspace/weights"
COMMIT="Update model"
api.create_repo(
    repo_id=REPO_ID,
    exist_ok=True,
    private=False,  # set False for public
)

# Upload the folder
api.upload_folder(
    folder_path=FOLDER_PATH,
    repo_id=REPO_ID,
    commit_message=COMMIT,
)