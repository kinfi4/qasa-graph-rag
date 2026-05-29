import base64
import json
import os
import tempfile

from google import genai


def create_genai_client() -> genai.Client:
    """Create a Google GenAI client.

    Two authentication modes are supported, in priority order:

    1. **Gemini Developer API (recommended for reproduction).** Set
       ``GOOGLE_API_KEY`` (or ``GEMINI_API_KEY``) to a key from
       https://aistudio.google.com/apikey.
    2. **Vertex AI.** Set ``GOOGLE_APPLICATION_CREDENTIALS_JSON`` to a
       base64-encoded service-account JSON; the client then talks to Vertex AI
       using the service account's ``project_id`` in ``us-central1``.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    credentials_b64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if credentials_b64:
        credentials_json = base64.b64decode(credentials_b64).decode("utf-8")
        credentials = json.loads(credentials_json)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(credentials, f)
            temp_path = f.name

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
        project_id = credentials.get("project_id")

        return genai.Client(vertexai=True, project=project_id, location="us-central1")

    raise ValueError(
        "No credentials found. Set GOOGLE_API_KEY (Gemini Developer API) or "
        "GOOGLE_APPLICATION_CREDENTIALS_JSON (Vertex AI service account). "
        "See .env.example."
    )
