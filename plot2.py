import json
import matplotlib.pyplot as plt
import os

def plot_loss_curves(log_file_path):
    """
    读取Hugging Face Trainer的日志文件并绘制训练和评估损失曲线。
    """
    # --- 1. 检查并加载日志文件 ---
    if not os.path.exists(log_file_path):
        print(f"错误: 日志文件未找到于 '{log_file_path}'")
        return

    with open(log_file_path, 'r') as f:
        log_history = json.load(f)

    # --- 2. 解析日志数据 ---
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        elif 'eval_loss' in entry and 'step' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])

    if not train_steps or not eval_steps:
        print("错误: 日志文件中缺少训练或评估数据。")
        return

    # --- 3. 绘制图形 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(train_steps, train_losses, label='Training Loss', color='royalblue', alpha=0.8)
    ax.plot(eval_steps, eval_losses, label='Evaluation Loss', color='darkorange', marker='o', linestyle='--')

    # --- 4. 标注最优点 ---
    min_eval_loss = min(eval_losses)
    min_eval_step = eval_steps[eval_losses.index(min_eval_loss)]
    ax.annotate(
        f'Best Model (Eval Loss: {min_eval_loss:.4f})',
        xy=(min_eval_step, min_eval_loss),
        xytext=(min_eval_step, min_eval_loss + 0.1 * (max(train_losses) - min(eval_losses))),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        ha='center'
    )

    # --- 5. 美化 & 保存 ---
    ax.set_title('Training and Evaluation Loss Curves', fontsize=16)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)

    output_plot_path = os.path.join(os.path.dirname(log_file_path), "training_history_plot.png")
    plt.savefig(output_plot_path, dpi=300)
    print(f"图形已保存至: {output_plot_path}")
    plt.show()


if __name__ == "__main__":
    LOG_FILE = "./output_query_rewriter_lora/training_log_history.json"
    plot_loss_curves(LOG_FILE)
