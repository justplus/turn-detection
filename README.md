# Turn Detection - 对话轮次检测模型

## 1. 介绍

Turn Detection（对话轮次检测）是一个用于人机对话系统中的关键技术，主要用于：
- **对话边界识别**：准确判断用户何时结束当前发言，避免对话系统过早或过晚响应
- **多轮对话管理**：在连续对话中识别每个对话轮次的开始和结束，提升对话体验
- **实时交互优化**：通过精准的轮次检测，实现更自然流畅的人机交互
- **语音助手增强**：为语音助手、客服机器人等应用提供更智能的对话控制

模型基于gemma3 270M模型进行微调，提供了完整的数据集和微调脚本。
效果媲美7B模型效果。


## 2. 主要特点

### 🔄 支持多轮对话
- 能够处理复杂的多轮对话场景
- 准确识别对话中的停顿、思考和真正的轮次结束
- 支持上下文感知的轮次判断

  支持多轮对话的重要性：
  ```
  user: 我们来个成语接龙吧？
  assistant: 那我先来，杞人忧天。该你了
  user: 天天向上
  ```
  非多轮对话下单一的"天天向上"是不完整的，但是放在上下文中则应该是完整的。

### 🚀 轻量化推理
- 模型参数仅270M，资源占用低
- 支持CPU推理，无需GPU即可部署
- 推理速度快，满足实时对话需求
- 适合边缘设备和资源受限环境

### 🌍 多语言支持
- 原生支持中文和英文对话检测
- 模型架构支持微调扩展到其他语言
- 跨语言泛化能力强

### 🛠️ 可定制化
- 提供完整的微调框架
- 支持针对特定领域和语言的定制训练
- 灵活的数据处理和训练流程

### 🙅‍♂️ 支持等待状态
- **0 (不完整)**：用户话语未说完，需要等待继续输入
- **1 (完整)**：用户话语表达完整，可以进行回复
- **2 (要求等待)**：用户要求暂停或打断AI回复

## 3. 微调过程

### 数据集构造
中文单轮和多轮数据：使用LLM合成
英文单轮和多轮数据：[turns-2k](https://huggingface.co/datasets/latishab/turns-2k/)数据集使用LLM扩展为多轮

### 微调训练
使用 `finetune.py` 进行模型微调：
```bash
pip install -r requirements.txt
python finetune.py
```

如果微调的过程中出现下面的错误，unsloth依赖的triton版本过高，需要卸载triton版本，重新安装triton-3.2.0版本
```bash
pip uninsatll triton
pip install triton==3.2.0
```
```plain text
torch._inductor.exc.InductorError: AttributeError: type object 'CompiledKernel' has no attribute 'launch_enter_hook'

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```


## 4. 模型效果

### 效果指标
中文准确率: 0.9591 (258/269)<br/>
英文准确率: 0.9654 (223/231)<br/>
总体准确率: 0.9620 (481/500)

### 性能指标
Nvidia T4单卡推理耗时: <100ms (P95<80ms)

## 5. 部署与推理
### 推理注意事项
- 中文场景使用中文的[system prompt](system_prompt_cn.txt)，英文场景使用英文的[system prompt](system_prompt_en.txt)
- wait场景在多轮对话中才有效，结合实际使用场景，训练集中wait场景均为多轮对话。
- 训练数据中未使用通用数据集进行配比训练，所以通用能力可能会有下降。如果需要通用能力请在当前数据集基础上添加通用数据集进行训练，通常做1:1配比即可。

### 模型权重
[justpluso/turn-detection](https://huggingface.co/justpluso/turn-detection)

国内访问huggingface遇到网络问题时，可以设置
```
# For Linux or MacOS
export HF_ENDPOINT=https://hf-mirror.com
```
或
```
# For Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 使用vLLM部署
```bash
# 启动HTTP API服务
vllm serve gemma3-270m-full-turn-detection --served-model-name=gemma3 --port 8000 --enable-prefix-caching --gpu-memory-utilization 0.8

# 调用API
curl -X POST http://localhost:8000/v1 \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64_encoded_audio"}'
```
也兼容openAI库。

### 使用
#### transformers库调用
```python
from inference import TurnDetector

# 初始化检测器
detector = TurnDetector(
    model_path="gemma3-270m-full-turn-detection",  # 模型路径
    device="auto"  # 自动选择设备，也可以指定"cpu"或"cuda"
)

# 方式1: 字符串格式输入
conversation_str = """user: 我们来成语接龙吧
assistant: 杞人忧天
user: 天天向上"""

result = detector.detect(conversation_str)
print(f"检测结果: {result}")  # 0-完整, 1-不完整, 2-要求等待

# 方式2: 消息列表格式输入
conversation_msgs = [
    {"role": "user", "content": "我们来成语接龙吧"},
    {"role": "assistant", "content": "杞人忧天"},
    {"role": "user", "content": "天天向上"}
]

result = detector.detect(conversation_msgs)
print(f"检测结果: {result}")

# 方式3: 获取详细结果
detailed_result = detector.detect_with_explanation(conversation_str)
print(f"状态码: {detailed_result['status_code']}")
print(f"状态名: {detailed_result['status_name']}")
print(f"说明: {detailed_result['description']}")

# 方式4: 批量检测
conversations = [
    "user: 我想要...",
    "user: 停",
    "user: 今天天气很好"
]

results = detector.detect_batch(conversations)
print(f"批量检测结果: {results}")  # [1, 2, 0]
```

#### 命令行使用
```bash
# 交互式模式
python inference.py --interactive

# 单次检测
python inference.py --input "user: 我想要..."

# 批量检测
python inference.py --input_file conversations.json --output_file results.json

# 指定设备和参数
python inference.py --device cpu --temperature 0.1 --interactive

# 演示示例
python inference.py
```


#### API服务部署
```bash
# 启动HTTP API服务
vllm serve gemma3-270m-full-turn-detection --gpu-memory-utilization 0.8 --enable-prefix-caching --served-model-name=gemma3-turn-detection --port 8080 

# 调用API
curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xx" \
  -d '{
    "model": "gemma3-turn-detection",
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "messages": [
      {
        "role": "system",
        "content": "你是一个专门分析对话状态的AI助手。请根据对话历史，判断用户最后说的话属于以下哪种状态：\n\n**状态定义：**\n- 0 (不完整)：用户的话语表达完整，意思清晰明确，不需要继续补充\n- 1 (完整)：用户的话语未说完，存在停顿、犹豫或明显的未完成表达\n- 2 (要求等待)：用户明确表示要打断或暂停AI的回复，要求AI停止说话或等待\n\n**判断标准：**\n\n**不完整(0)的特征：**\n- 句子突然中断，没有完整表达意思\n- 包含停顿词：如"嗯"、"那个"、"就是"、"呃"等\n- 语句结构不完整，明显还有后续内容\n- 例如："我想要..."、"关于这个问题，嗯..."、"山字怎么"\n\n**完整(1)的特征：**\n- 句子结构完整，语法正确\n- 表达了清晰的意图或完整的信息\n- 没有明显的停顿词或未完成标记\n- 例如："我想去北京旅游"、"今天天气很好"、"谢谢你的帮助"\n\n**要求等待(2)的特征：**\n- 明确的打断指令：如"停"、"等等"、"暂停"、"闭嘴"\n- 礼貌的暂停请求：如"稍等"、"等一下"、"先别说"\n- 表达需要时间思考：如"让我想想"、"我需要安静"\n- 表达不耐烦：如"够了"、"太多了"、"别说了"\n- 英文打断：如"Stop"、"Wait"、"Hold on"、"Shut up"、"Enough"\n\n\n**输出格式：**\n你只能输出[0、1、2]中的其中一个数字，不要输出其他的。"
      },
      {
        "role": "user", 
        "content": "请分析以下对话中用户最后说的话：\nuser: 我们来成语接龙吧\nassistant: 杞人忧天\nuser: 停"
      }
    ]
  }'
```

## More
- 可以基于提供的训练脚本新增其他语种的语料进行继续微调。每个语种在200条数据即可达到比较好的效果
- 模型可以量化以进一步降低资源占用，提升推理效率。


## 致谢
- [Unsloth](https://unsloth.ai/): 优秀的微调框架
- [Gemma3](https://deepmind.google/models/gemma/gemma-3/): 优秀的开源模型权重
- [ten-turn-detection](https://github.com/TEN-framework/ten-turn-detection): 参考了其wait数据集，并对比了其模型效果

---

## License
[This project is Apache 2.0 licensed with certain conditions.](LICENSE)
