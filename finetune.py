#!/usr/bin/env python3
from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
from transformers import TextStreamer
import json
import re
import numpy as np
import os
from transformers import TrainerCallback
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"


# 读取改进的system prompt
def load_system_prompt_cn():
    with open("system_prompt_cn.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

def load_system_prompt_en():
    with open("system_prompt_en.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

def detect_language(text):
    """检测文本语言（中文或英文）"""
    import re
    
    # 移除用户标记，只分析实际对话内容
    clean_text = text.replace("user: ", "").replace("assistant: ", "")
    
    # 统计中文字符数量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean_text))
    
    # 统计英文字母数量
    english_chars = len(re.findall(r'[a-zA-Z]', clean_text))
    
    # 统计总字符数（排除空格和标点）
    total_chars = chinese_chars + english_chars
    
    if total_chars == 0:
        return "en"  # 默认返回英文
    
    # 如果中文字符占比超过30%，认为是中文
    chinese_ratio = chinese_chars / total_chars
    
    if chinese_ratio > 0.3:
        return "zh"
    else:
        return "en"

def convert_to_chatml(example):
    # 检测语言
    language = detect_language(example["input"])
    
    if language == "zh":
        # 中文语料使用中文prompt
        system_prompt = load_system_prompt_cn()
        user_prompt = f'请分析以下对话中用户最后说的话：\n{example["input"]}'
    else:
        # 英文语料使用英文prompt  
        system_prompt = load_system_prompt_en()
        user_prompt = f'Please analyze the user\'s last utterance in the following conversation:\n{example["input"]}'
    
    return {
        "conversations": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": str(example["target"])}  # 将target转换为字符串
        ]
    }

class GenerationEvaluationCallback(TrainerCallback):
    """自定义回调类，用于在评估阶段进行真正的生成式评估"""
    
    def __init__(self, eval_dataset, tokenizer):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        # 预加载系统提示词
        self.zh_system_prompt = load_system_prompt_cn()
        self.en_system_prompt = load_system_prompt_en()
        
    def on_evaluate(self, args, state, control, model, **kwargs):
        """在评估时进行生成式评估"""         
        print("\n" + "="*50)
        print(f"开始生成式评估... (步数: {state.global_step})")
        print("="*50)
        
        # 准备统计指标的变量
        zh_correct = 0
        zh_total = 0
        en_correct = 0
        en_total = 0
        total_samples = 0
        
        # 只评估前几个样本以节省时间
        eval_samples = len(self.eval_dataset)
        
        for i in range(eval_samples):
            sample = self.eval_dataset[i]
            
            # 检测语言并选择相应的system prompt和用户提示
            language = detect_language(sample['input'])
            
            if language == "zh":
                system_prompt = self.zh_system_prompt
                user_content = f"请分析以下对话中用户最后说的话：\n{sample['input']}"
            else:
                system_prompt = self.en_system_prompt
                user_content = f"Please analyze the user's last utterance in the following conversation:\n{sample['input']}"
            
            # 构建输入消息
            messages = [
                {'role': 'system', 'content': system_prompt},
                {"role": 'user', 'content': user_content}
            ]
            
            try:
                # 应用chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False
                )
                
                # 生成响应
                inputs = self.tokenizer([text], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=64,
                        temperature=1.0,
                        top_p=0.95,
                        top_k=64,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                # 解码响应
                response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                
                # 提取生成的部分
                match = re.search(r"model\n(.*?)(?:<end_of_turn>|$)", response, re.S)
                if match:
                    generated_text = match.group(1).strip()
                else:
                    generated_text = ""
                
                # 调试信息：显示生成的内容
                # print(f"样本 {i+1}:")
                # print(f"  输入: {sample['input']}")
                # print(f"  生成: {generated_text}")
                # print(f"  目标: {sample['target']}")
                
                # 直接比较生成的标签和目标标签
                try:
                    # 尝试将生成的文本转换为整数
                    predicted_label = int(generated_text.strip())
                    target_label = sample['target']
                    is_correct = predicted_label == target_label
                    
                    # 根据语言分别统计
                    if language == "zh":
                        zh_total += 1
                        if is_correct:
                            zh_correct += 1
                    else:
                        en_total += 1
                        if is_correct:
                            en_correct += 1
                    
                except (ValueError, TypeError):
                    # 如果无法解析为整数，算作预测错误，但仍要统计到总数中
                    if language == "zh":
                        zh_total += 1
                    else:
                        en_total += 1
                
                total_samples += 1
                
            except Exception as e:
                # print(f"评估样本 {i} 时出错: {e}")
                continue
        
        # 计算最终指标
        if total_samples > 0:
            # 打印结果
            print(f"生成式评估结果 (基于 {total_samples} 个样本):")
            
            if zh_total > 0:
                zh_accuracy = zh_correct / zh_total
                print(f"  中文准确率: {zh_accuracy:.4f} ({zh_correct}/{zh_total})")
            else:
                print(f"  中文准确率: 无中文样例")
            
            if en_total > 0:
                en_accuracy = en_correct / en_total
                print(f"  英文准确率: {en_accuracy:.4f} ({en_correct}/{en_total})")
            else:
                print(f"  英文准确率: 无英文样例")
            
            overall_accuracy = (zh_correct + en_correct) / total_samples
            print(f"  总体准确率: {overall_accuracy:.4f} ({zh_correct + en_correct}/{total_samples})")
            print(f"  语言分布: 中文 {zh_total}条, 英文 {en_total}条")
            print("="*50)
            
            # 将指标添加到日志中，但不返回字典（避免与control对象冲突）
            if hasattr(control, 'should_log'):
                control.should_log = True
            
            # 将指标存储到state中以便记录
            if hasattr(state, 'log_history'):
                metrics = {
                    "eval_gen_accuracy": overall_accuracy,
                    "eval_gen_zh_accuracy": zh_accuracy if zh_total > 0 else 0,
                    "eval_gen_en_accuracy": en_accuracy if en_total > 0 else 0,
                }
                # 可以选择将指标添加到日志历史中
                # state.log_history.append(metrics)
        
        # TrainerCallback的on_evaluate方法不应该返回字典
        return None

def train_model():
    # 加载数据集
    dataset = load_dataset("json", data_files="SFT.jsonl")["train"]
    dataset = dataset.shuffle(seed=42)
    
    test_size = 500
    eval_size = 100
    train_dataset = dataset.select(range(test_size, len(dataset)))
    test_dataset = dataset.select(range(test_size))
    eval_dataset = test_dataset.select(range(eval_size))
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 转换数据格式 - 使用改进的system prompt
    print("转换数据格式...")
    train_dataset = train_dataset.map(convert_to_chatml)
    test_dataset = test_dataset.map(convert_to_chatml)
    eval_dataset = eval_dataset.map(convert_to_chatml)
    
    # 加载模型
    max_seq_length = 1024
    model, tokenizer = FastModel.from_pretrained(
        model_name = "../gemma3-270m/model",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        load_in_8bit = False,
        full_finetuning = False,
    )
    
    model = FastModel.get_peft_model(
        model,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        r = 16, 
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False, 
        loftq_config = None,
    )
    
    # 设置chat template
    tokenizer = get_chat_template(tokenizer, chat_template = "gemma3")
    
    def formatting_prompts_func(examples):
       convos = examples["conversations"]
       texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
       return { "text" : texts, }
    
    train_dataset = train_dataset.map(formatting_prompts_func, batched = True)
    test_dataset = test_dataset.map(formatting_prompts_func, batched = True)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True)
    
    
    # 创建生成式评估回调（使用原始数据集，不是格式化后的）
    generation_callback = GenerationEvaluationCallback(
        eval_dataset=test_dataset.select(range(eval_size)), 
        tokenizer=tokenizer
    )
    
    # 训练配置 - 更多步数和更小学习率
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 8,  # 减小batch size以适应更长序列
            gradient_accumulation_steps = 1,  # 增加梯度累积
            warmup_steps = 20,
            # max_steps = 100,
            num_train_epochs = 1,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir="outputs",
            report_to = "none",
            eval_strategy = "steps",
            save_strategy = "steps",
            eval_steps = 50,  # 减少评估频率，让模型有更多时间学习
            save_steps = 100,
            # 为生成式任务添加必要配置
            label_names = ["labels"],
            prediction_loss_only = False,
            # load_best_model_at_end = True, 
        ),
        # 对于生成式任务，我们不使用compute_metrics，而是依赖自定义回调
        callbacks = [generation_callback],  # 添加生成式评估回调
    )
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )
    
    # 开始训练
    print("开始训练...")
    trainer_stats = trainer.train()
    print("训练完成！")
    
    return model, tokenizer, test_dataset

def test_model(model, tokenizer, test_dataset):
    """测试改进后的模型"""
    print("\n" + "="*60)
    print("测试改进后的模型")
    print("="*60)
    
    # 预加载系统提示词
    zh_system_prompt = load_system_prompt_cn()
    en_system_prompt = load_system_prompt_en()

    fs = open('evaluate_result.txt', 'w', encoding='utf-8')
    
    # 分别统计中英文的准确率
    zh_correct = 0
    zh_total = 0
    en_correct = 0
    en_total = 0
    total_predictions = 0
    
    for i, test_input in enumerate(test_dataset, 1):
        print(f"\n测试 {i}:")
        print(f"输入: {test_input['input']}")
        
        # 检测语言并选择相应的system prompt和用户提示
        language = detect_language(test_input['input'])
        
        if language == "zh":
            system_prompt = zh_system_prompt
            user_content = f"请分析以下对话中用户最后说的话：\n{test_input['input']}"
            print(f"语言: 中文")
        else:
            system_prompt = en_system_prompt  
            user_content = f"Please analyze the user's last utterance in the following conversation:\n{test_input['input']}"
            print(f"语言: 英文")
        
        messages = [
            {'role': 'system','content': system_prompt},
            {"role" : 'user', 'content' : user_content}
        ]
        
        try:
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True,
                tokenize=False
            )

            from transformers import TextStreamer
            outputs = model.generate(
                **tokenizer([text], return_tensors = "pt").to("cuda"),
                max_new_tokens = 64,
                temperature = 1.0, top_p = 0.95, top_k = 64,
                streamer = TextStreamer(tokenizer, skip_prompt = True),
            )
            response = tokenizer.batch_decode(outputs)

            match = re.search(r"<start_of_turn>model\n(.*?)<end_of_turn>", response[0], re.S)
            if match:
                result = match.group(1).strip()

                print(f"输出: {result}")
                print(f"期望: {test_input['target']}")
                
                # 评估准确率
                try:
                    predicted_label = int(result.strip())
                    is_correct = predicted_label == test_input['target']
                    
                    # 根据语言分别统计
                    if language == "zh":
                        zh_total += 1
                        if is_correct:
                            zh_correct += 1
                    else:
                        en_total += 1
                        if is_correct:
                            en_correct += 1
                    
                    if is_correct:
                        print("✅ 预测正确")
                    else:
                        print("❌ 预测错误")
                        
                except ValueError:
                    print("❌ 输出格式错误")
                    # 格式错误也要统计到总数中
                    if language == "zh":
                        zh_total += 1
                    else:
                        en_total += 1
                
                total_predictions += 1
                print("="*60)

                fs.write(f"输入: {test_input['input']}\n")
                fs.write(f"输出: {result}\n")
                fs.write(f"期望: {test_input['target']}\n")
                fs.write("="*60 + "\n")
                
        except Exception as e:
            print(f"❌ 生成失败: {e}")
    
    fs.close()
    
    # 打印分语言的准确率统计
    print(f"\n{'='*60}")
    print("准确率统计")
    print("="*60)
    
    if zh_total > 0:
        zh_accuracy = zh_correct / zh_total
        print(f"中文准确率: {zh_accuracy:.4f} ({zh_correct}/{zh_total})")
    else:
        print("中文准确率: 无中文测试样例")
    
    if en_total > 0:
        en_accuracy = en_correct / en_total
        print(f"英文准确率: {en_accuracy:.4f} ({en_correct}/{en_total})")
    else:
        print("英文准确率: 无英文测试样例")
    
    if total_predictions > 0:
        overall_accuracy = (zh_correct + en_correct) / total_predictions
        print(f"总体准确率: {overall_accuracy:.4f} ({zh_correct + en_correct}/{total_predictions})")
        print(f"语言分布: 中文 {zh_total}条 ({zh_total/total_predictions*100:.1f}%), 英文 {en_total}条 ({en_total/total_predictions*100:.1f}%)")
    
    print("="*60)

def save_models(model, tokenizer):
    """保存改进后的模型"""
    print("\n" + "="*60)
    print("保存改进后的模型")
    print("="*60)
    
    # 保存LoRA adapter
    print("保存LoRA adapter...")
    model.save_pretrained("gemma3-270m-lora-turn-detection")
    tokenizer.save_pretrained("gemma3-270m-lora-turn-detection")
    
    # 保存合并后的完整模型
    print("保存合并后的完整模型...")
    model.save_pretrained_merged("gemma3-270m-full-turn-detection", tokenizer, save_method="merged_16bit")
    # 确保与模型同目录的tokenizer也保存（包含chat_template与特殊token设置）
    tokenizer.save_pretrained("gemma3-270m-full-turn-detection")
    
    print("改进后的模型保存完成！")

def main():
    print("="*60)
    
    # 1. 重新训练
    model, tokenizer, test_dataset = train_model()
    
    # 2. 测试改进后的模型
    test_model(model, tokenizer, test_dataset)
    
    # 3. 保存改进后的模型
    save_models(model, tokenizer)
    
    print("\n" + "="*60)
    print("改进版模型训练完成！")
    print("="*60)

if __name__ == "__main__":
    main()