import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "EleutherAI/gpt-neo-125M"
local_save_path = "./local_gpt_neo_125M"

# 设置环境变量
os.environ["TRANSFORMERS_MIRROR"] = "https://hf-mirror.com"

# 手动指定镜像站 URL 格式
def custom_hf_hub_download(repo_id, filename, **kwargs):
    base_url = os.environ.get("TRANSFORMERS_MIRROR")
    url = f"{base_url}/{repo_id}/resolve/main/{filename}"
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename, **kwargs, url=url)

from transformers.utils.hub import cached_file
original_cached_file = cached_file

def custom_cached_file(path_or_repo_id, filename=None, **kwargs):
    if isinstance(path_or_repo_id, str) and "/" in path_or_repo_id:
        return custom_hf_hub_download(path_or_repo_id, filename, **kwargs)
    return original_cached_file(path_or_repo_id, filename, **kwargs)

from transformers.utils import hub
hub.cached_file = custom_cached_file

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_save_path)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_save_path)

tokenizer.save_pretrained(local_save_path)
model.save_pretrained(local_save_path)

print(f"模型和分词器已成功保存到 {local_save_path}")