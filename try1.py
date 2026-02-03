from transformers import AutoModel

model_path = "./bge-large-zh-v1.5-local"
output_path = "./bge-large-zh-v1.5-local-safetensors"

# 加载原始模型（bin）
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

# 保存为 safetensors
model.save_pretrained(output_path, safe_serialization=True)

print(f"模型已保存为 safetensors: {output_path}")

