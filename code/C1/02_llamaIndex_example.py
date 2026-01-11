import os
# os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.deepseek import DeepSeek
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels

load_dotenv()

# Settings.llm = DeepSeek(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"))
Settings.llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_PLUS,
    api_key=os.getenv("QWEN_API_KEY"),  # 也可以不写，让它自动读环境变量
)
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

docs = SimpleDirectoryReader(input_files=["../../data/C1/markdown/easy-rl-chapter1.md"]).load_data()

index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine()

print(query_engine.get_prompts())

print(query_engine.query("文中举了哪些例子?"))