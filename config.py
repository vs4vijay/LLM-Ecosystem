import dotenv
import os

dotenv.load_dotenv()


class Config:
    openai_api_type = "azure"
    openai_api_version = "2023-05-15"
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_model_name = os.environ.get("OPENAI_MODEL_NAME")
    openai_deployment_name = os.environ.get("OPENAI_DEPLOYMENT_NAME")


config = Config()
