# ========== 0. 导入所需库 ==========
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import LoraConfig, TaskType, get_peft_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ========== 1. 加载我们新生成的微调数据集 ==========
# ✅ 使用我们新合成的数据集
DATA_FILE = "synthetic_query_rewriter_dataset_robust_1k.jsonl"
print(f"--- 步骤 1: 正在加载 {DATA_FILE} 文件 ---")
dataset = load_dataset('json', data_files=DATA_FILE, split='train')

# ✅ 划分训练集和验证集 (95% 训练, 5% 验证)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]
print("数据集加载并划分成功！")
print(f"训练集样本数: {len(train_dataset)}")
print(f"验证集样本数: {len(eval_dataset)}")


# ========== 2. 初始化分词器和模型 ==========
model_name_or_path = '/root/autodl-tmp/legal_finetune/deepseek'
print(f"--- 步骤 2: 正在从本地路径初始化: {model_name_or_path} ---")

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=False,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ========== 3. 数据预处理函数 (核心改动) ==========
# 我们需要将 instruction, input, output 拼接成一个完整的prompt
def format_and_tokenize(example):
    MAX_LENGTH = 768 # 对于这个任务，768的长度足够了

    # 构建一个标准的指令跟随格式
    prompt_template = (
        "你是一个法律查询助手。请分析用户的法律问题，并将其转换为结构化的JSON对象，"
        "包含用于关键词搜索的'keywords_for_search'和用于向量搜索的'query_for_vector_search'。\n\n"
        "### 用户问题:\n{input}\n\n### JSON输出:\n{output}"
    )
    
    text = prompt_template.format(
        input=example['input'],
        output=example['output']
    ) + tokenizer.eos_token # 在末尾添加结束符

    tokenized = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    # 在这个任务中，我们让模型预测整个序列，所以labels就是input_ids的拷贝
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

print("--- 步骤 3: 正在对数据集进行格式化和分词处理 ---")
tokenized_train_dataset = train_dataset.map(format_and_tokenize, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(format_and_tokenize, remove_columns=eval_dataset.column_names)
print("数据处理完成！")


# ========== 4. 加载基础模型 ==========
print(f"--- 步骤 4: 正在加载基础模型... ---")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype="auto",
    trust_remote_code=True,
    device_map="auto"
)
model.enable_input_require_grads()


# ========== 5. LoRA 配置 ==========
print("--- 步骤 5: 正在配置LoRA ---")
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # 适配DeepSeek-LLM-7B模型的模块名
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


# ========== 6. 训练参数 ==========
print("--- 步骤 6: 正在设置训练参数 ---")
# ✅ 使用新的输出目录，避免覆盖
OUTPUT_DIR = "./output_query_rewriter_lora"
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8, # 实际 batch_size = 4 * 8 = 32
    logging_steps=5, # 更频繁地记录日志
    num_train_epochs=5,
    save_strategy="steps",
    save_steps=15, # 根据数据集大小调整，更频繁地保存和评估
    learning_rate=2e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    save_safetensors=True,
    # ✅ 启用评估和早停
    evaluation_strategy="steps",
    eval_steps=15, # 与save_steps保持一致
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

# ✅ 定义早停回调
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=5, # 稍微增加耐心，因为初期损失可能波动
    early_stopping_threshold=0.005,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    callbacks=[early_stopping_callback],
)

# ========== 7. 启动训练 ==========
print("--- 步骤 7: 所有准备就绪，即将开始查询重写模型的LoRA微调！---")
trainer.train()

# ========== 8. 保存最终模型和训练日志 ==========
print("--- 步骤 8: 训练结束，正在保存最终的最佳模型 ---")
final_model_path = f"{OUTPUT_DIR}/final_model"
trainer.save_model(final_model_path)
print(f"\n--- 🎉 查询重写模型微调完成！最佳模型已保存在: {final_model_path} ---")

# ✅ 保存训练历史记录到新的输出目录
log_history_path = f"{OUTPUT_DIR}/training_log_history.json"
with open(log_history_path, "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)
print(f"训练日志已保存至: {log_history_path}")
