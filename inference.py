#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn Detection 推理模块
使用 transformers 库进行对话轮次检测推理
"""

import json
import argparse
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 读取改进的system prompt
def load_system_prompt_cn():
    with open("system_prompt_cn.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

def load_system_prompt_en():
    with open("system_prompt_en.txt", "r", encoding="utf-8") as f:
        return f.read().strip()

class TurnDetector:
    """对话轮次检测器"""
    
    def __init__(self, model_path: str = "justpluso/turn-detection", device: str = "auto"):
        """
        初始化检测器
        
        Args:
            model_path: 模型路径或HuggingFace模型名称
            device: 设备类型，"auto"自动选择，"cpu"使用CPU，"cuda"使用GPU
        """
        self.model_path = model_path
        
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # 加载tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
        # 系统提示词 - 中文
        self.system_prompt_zh = load_system_prompt_cn()

        # 系统提示词 - 英文
        self.system_prompt_en = load_system_prompt_en()

    def _detect_language(self, text: str) -> str:
        """
        检测文本主要语言
        
        Args:
            text: 输入文本
            
        Returns:
            语言代码：'zh' 或 'en'
        """
        # 简单的语言检测逻辑
        clean_text = text.replace("user: ", "").replace("assistant: ", "")
        chinese_chars = len([c for c in clean_text if '\u4e00' <= c <= '\u9fff'])
        total_chars = len([c for c in clean_text if c.isalnum()])
        
        if total_chars == 0:
            return 'zh'  # 默认中文
            
        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.3 else 'en'
    
    def _format_conversation(self, conversation: Union[str, List[Dict[str, str]]]) -> str:
        """
        格式化对话内容
        
        Args:
            conversation: 对话内容，可以是字符串或消息列表
            
        Returns:
            格式化后的对话字符串
        """
        if isinstance(conversation, str):
            return conversation
            
        formatted_lines = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                formatted_lines.append(f"{role}: {content}")
                
        return "\n".join(formatted_lines)

    def detect(self, conversation: Union[str, List[Dict[str, str]]], temperature: float = 0.1) -> int:
        """
        检测对话轮次状态
        
        Args:
            conversation: 对话内容
            temperature: 生成温度
            
        Returns:
            状态码：0-完整，1-不完整，2-要求等待
        """
        # 格式化对话
        formatted_conv = self._format_conversation(conversation)
        
        # 检测语言并选择合适的system prompt
        language = self._detect_language(formatted_conv)
        system_prompt = self.system_prompt_zh if language == 'zh' else self.system_prompt_en
        
        # 构建用户提示词
        if language == 'zh':
            user_content = f"请分析以下对话中用户最后说的话：\n{formatted_conv}"
        else:
            user_content = f"Please analyze the user's last message in the following conversation:\n{formatted_conv}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # 应用聊天模板
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # 只需要生成一个数字
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # 提取数字结果
        try:
            # 查找响应中的数字
            for char in response:
                if char in ['0', '1', '2']:
                    return int(char)
            # 如果没有找到有效数字，返回默认值
            print(f"Warning: No valid result found in response: '{response}', returning 0")
            return 0
        except Exception as e:
            print(f"Error parsing response: '{response}', error: {e}")
            return 0

    def detect_batch(self, conversations: List[Union[str, List[Dict[str, str]]]], temperature: float = 0.1) -> List[int]:
        """
        批量检测对话轮次状态
        
        Args:
            conversations: 对话内容列表
            temperature: 生成温度
            
        Returns:
            状态码列表
        """
        results = []
        for conv in conversations:
            result = self.detect(conv, temperature)
            results.append(result)
        return results

    def detect_with_explanation(self, conversation: Union[str, List[Dict[str, str]]], temperature: float = 0.1) -> Dict[str, Any]:
        """
        检测对话轮次状态并返回详细信息
        
        Args:
            conversation: 对话内容
            temperature: 生成温度
            
        Returns:
            包含状态码和说明的字典
        """
        result = self.detect(conversation, temperature)
        
        status_map = {
            0: "完整",
            1: "不完整", 
            2: "要求等待"
        }
        
        description_map = {
            0: "用户的话语表达完整，意思清晰明确，不需要继续补充",
            1: "用户的话语未说完，存在停顿、犹豫或明显的未完成表达",
            2: "用户明确表示要打断或暂停AI的回复，要求AI停止说话或等待"
        }
        
        return {
            "status_code": result,
            "status_name": status_map.get(result, "未知"),
            "description": description_map.get(result, "未知状态"),
            "conversation": self._format_conversation(conversation)
        }


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Turn Detection 推理工具")
    parser.add_argument("--model_path", type=str, default="gemma3-270m-full-turn-detection",
                       help="模型路径或HuggingFace模型名称")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="设备类型")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="生成温度")
    parser.add_argument("--input", type=str,
                       help="输入对话内容（字符串格式）")
    parser.add_argument("--input_file", type=str,
                       help="输入对话文件（JSON格式）")
    parser.add_argument("--output_file", type=str,
                       help="输出结果文件")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式模式")
    
    args = parser.parse_args()
    
    # 初始化检测器
    detector = TurnDetector(args.model_path, args.device)
    
    if args.interactive:
        # 交互式模式
        print("=== Turn Detection 交互式模式 ===")
        print("输入格式: user: 你好\\nassistant: 你好！\\nuser: 我想...")
        print("输入 'quit' 退出")
        print()
        
        while True:
            try:
                user_input = input("请输入对话内容: ").strip()
                if user_input.lower() == 'quit':
                    break
                    
                if not user_input:
                    continue
                    
                result = detector.detect_with_explanation(user_input, args.temperature)
                print(f"检测结果: {result['status_code']} - {result['status_name']}")
                print(f"说明: {result['description']}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n退出交互式模式")
                break
            except Exception as e:
                print(f"错误: {e}")
                
    elif args.input:
        # 单次检测
        result = detector.detect_with_explanation(args.input, args.temperature)
        
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {args.output_file}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
    elif args.input_file:
        # 批量检测
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            conversations = data
        else:
            conversations = [data]
            
        results = []
        for i, conv in enumerate(conversations):
            result = detector.detect_with_explanation(conv, args.temperature)
            results.append(result)
            print(f"处理进度: {i+1}/{len(conversations)}")
            
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {args.output_file}")
        else:
            print(json.dumps(results, ensure_ascii=False, indent=2))
            
    else:
        # 演示示例
        print("=== Turn Detection 演示 ===")
        
        examples = [
            "user: 我们来成语接龙吧\nassistant: 杞人忧天\nuser: 天天向上",
            "user: 我想要...",
            "user: 停",
            "user: 今天天气很好",
            "user: 关于这个问题，嗯...",
            "user: 等等，让我想想"
        ]
        
        for example in examples:
            print(f"\n输入: {repr(example)}")
            result = detector.detect_with_explanation(example, args.temperature)
            print(f"结果: {result['status_code']} - {result['status_name']}")
            print(f"说明: {result['description']}")


if __name__ == "__main__":
    main()
