import json
import matplotlib.pyplot as plt
import argparse

def plot_loss_curve(log_file, output_image):
    """
    从Hugging Face Trainer的日志历史文件中读取数据并绘制训练和验证损失曲线。
    """
    try:
        with open(log_file, 'r') as f:
            log_history = json.load(f)
    except FileNotFoundError:
        print(f"错误: 日志文件 '{log_file}' 未找到。请确保训练已完成且文件路径正确。")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析日志文件 '{log_file}'。文件可能已损坏或格式不正确。")
        return

    # 从日志中提取训练和验证数据
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []

    for log in log_history:
        if 'loss' in log: # 训练日志
            train_steps.append(log['step'])
            train_losses.append(log['loss'])
        if 'eval_loss' in log: # 验证日志
            eval_steps.append(log['step'])
            eval_losses.append(log['eval_loss'])

    if not train_steps:
        print("警告: 在日志中未找到训练损失数据。")
        return
        
    # 创建图表
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # 绘制训练损失曲线
    ax.plot(train_steps, train_losses, label='Training Loss', color='dodgerblue', marker='o', linestyle='-', markersize=4)

    # 绘制验证损失曲线（如果存在）
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label='Validation Loss', color='tomato', marker='s', linestyle='--', markersize=4)
        # 找到最低验证损失点并标记
        min_eval_loss = min(eval_losses)
        min_eval_step = eval_steps[eval_losses.index(min_eval_loss)]
        ax.axvline(x=min_eval_step, color='limegreen', linestyle=':', linewidth=2, label=f'Best Model (Step {min_eval_step})')
        ax.annotate(f'Lowest Loss: {min_eval_loss:.4f}',
                    xy=(min_eval_step, min_eval_loss),
                    xytext=(min_eval_step + 10, min_eval_loss + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.7))


    # 设置图表标题和标签
    ax.set_title('Training and Validation Loss Curve', fontsize=16, fontweight='bold')
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    
    # 优化刻度显示
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_image)
    print(f"🎉 训练曲线图已成功保存到: {output_image}")


if __name__ == '__main__':
    # 使用 argparse 允许从命令行指定文件路径
    parser = argparse.ArgumentParser(description="从Hugging Face Trainer日志绘制损失曲线。")
    parser.add_argument(
        "--log_file",
        type=str,
        default="./output_deepseek_legal_lora_v2/training_log_history.json",
        help="训练日志历史文件的路径 (training_log_history.json)。"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="./output_deepseek_legal_lora_v2/loss_curve.png",
        help="输出的损失曲线图片文件路径。"
    )

    args = parser.parse_args()
    plot_loss_curve(args.log_file, args.output_image)
