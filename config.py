import dotenv
import os

dotenv.load_dotenv()


class Config:
    openai_api_type = "azure"
    azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    azure_openai_api_base = os.environ.get("AZURE_OPENAI_API_BASE")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model_name = os.environ.get("OPENAI_MODEL_NAME")
    openai_deployment_name = os.environ.get("OPENAI_DEPLOYMENT_NAME")


config = Config()
