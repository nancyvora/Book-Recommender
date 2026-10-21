"""Upload the web app to a Hugging Face Space.

Usage (from the project root, after `hf auth login`):
    python deploy/deploy_space.py <your-username>/<space-name>

Only the app, the search code and its data are uploaded. The OpenAI key is NOT uploaded:
add it yourself as a secret in the Space settings (see the instructions printed at the end).
"""
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = ["app.py", "recommender.py", "cover-not-found.jpg", "data/processed/books_with_emotions.csv"]
FOLDERS = ["data/chroma_db"]
GITHUB_URL = "https://github.com/nancyvora/Book-Recommender"

SPACE_README = f"""---
title: Find Your Next Read
emoji: 📚
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Book recommendations matched by meaning
---

# Find Your Next Read

Describe the book you're in the mood for, and get recommendations matched by meaning, not keywords.

Built with OpenAI embeddings, a Chroma vector database and Gradio. The code, notebooks and
evaluation are on GitHub: {GITHUB_URL}
"""


def space_requirements() -> str:
    # Spaces installs Gradio itself (from sdk_version), and the API server isn't used there
    skip = ("gradio", "fastapi", "uvicorn")
    lines = (ROOT / "requirements-app.txt").read_text().splitlines()
    return "\n".join(l for l in lines if l and not l.startswith("#") and not l.startswith(skip)) + "\n"


def main(repo_id: str) -> None:
    api = HfApi()
    print(f"Signed in as {api.whoami()['name']}")
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        staging = Path(staging)
        for file in FILES:
            (staging / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / file, staging / file)
        for folder in FOLDERS:
            shutil.copytree(ROOT / folder, staging / folder)
        (staging / "README.md").write_text(SPACE_README, encoding="utf-8")
        (staging / "requirements.txt").write_text(space_requirements(), encoding="utf-8")

        print("Uploading:", sorted(str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file()))
        api.upload_folder(folder_path=staging, repo_id=repo_id, repo_type="space",
                          commit_message="Deploy book recommender")

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"""
Uploaded to {url}

Finish setup in the Space's Settings tab ({url}/settings):
  1. Variables and secrets -> New secret -> Name: OPENAI_API_KEY, Value: your OpenAI key
  2. Space hardware -> choose ZeroGPU (free)
The Space rebuilds automatically; the first build takes a few minutes.""")


if __name__ == "__main__":
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit(__doc__)
    main(sys.argv[1])
