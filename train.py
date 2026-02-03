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
    EarlyStoppingCallback # 导入早停回调
)
from peft import LoraConfig, TaskType, get_peft_model
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ========== 1. 加载您处理好的法律数据集 ==========
# ✅ 使用新的数据集
DATA_FILE = "dataset_final_moreversion.jsonl"
print(f"--- 步骤 1: 正在加载 {DATA_FILE} 文件 ---")
dataset = load_dataset('json', data_files=DATA_FILE, split='train')

# ✅ 新增：划分训练集和验证集，防止过拟合
dataset = dataset.train_test_split(test_size=0.05, seed=42) # 95% 训练, 5% 验证
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
    # 大部分时候，`eos_token` 作为 `pad_token` 是一个合理的选择
    tokenizer.pad_token = tokenizer.eos_token

# ========== 3. 数据预处理函数 ==========
def process_func(example):
    MAX_LENGTH = 1024 # 根据您的模型和数据调整
    tokenized = tokenizer(
        example['text'],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

print("--- 步骤 3: 正在对数据集进行分词和标签化处理 ---")
tokenized_train_dataset = train_dataset.map(process_func, remove_columns=train_dataset.column_names)
tokenized_eval_dataset = eval_dataset.map(process_func, remove_columns=eval_dataset.column_names)
print("数据处理完成！")


# ========== 4. 加载模型 ==========
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
OUTPUT_DIR = "./output_deepseek_legal_lora_v2"
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # 根据您的显存大小调整
    gradient_accumulation_steps=8, # 实际 batch_size = 4 * 8 = 32
    logging_steps=10,
    num_train_epochs=5, # 设置一个相对较大的epoch数，让早停机制自动决定何时停止
    save_strategy="steps",
    save_steps=50, # 每50步保存一次模型
    learning_rate=2e-5, # 为微调任务选择一个较小的学习率
    save_on_each_node=True,
    gradient_checkpointing=True,
    save_safetensors=True,

    # ✅ 新增：启用评估和早停
    evaluation_strategy="steps",          # 每N步在验证集上评估一次
    eval_steps=50,                        # 与save_steps保持一致，每50步评估一次
    load_best_model_at_end=True,          # 训练结束后加载最佳模型
    metric_for_best_model="loss",         # 以验证集损失作为最佳模型的评判标准
    greater_is_better=False,              # 损失越小越好
)

# ✅ 新增：定义早停回调
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3, # 如果验证损失连续3次评估都没有改善，则停止训练
    early_stopping_threshold=0.01, # 改善必须超过这个阈值才算数
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset, # ✅ 传入验证集
    data_collator=data_collator,
    callbacks=[early_stopping_callback], # ✅ 应用早停回调
)

# ========== 7. 启动训练 ==========
print("--- 步骤 7: 所有准备就绪，即将开始LoRA微调！---")
trainer.train()

# ========== 8. 保存最终模型 ==========
print("--- 步骤 8: 训练结束，正在保存最终的最佳模型 ---")
# 保存最终的LoRA适配器
final_model_path = f"{OUTPUT_DIR}/final_model"
trainer.save_model(final_model_path)
print(f"\n--- 🎉 微调训练完成！最佳模型已保存在: {final_model_path} ---")

# 保存训练历史记录，以便绘图
with open(f"{OUTPUT_DIR}/training_log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=4)
