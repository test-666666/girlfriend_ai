"""
智能对话伴侣训练与交互系统
版本：2.1
功能特性：
1. 实时模型热更新机制
2. 多轮对话上下文管理
3. 情感感知回复生成
4. 用户反馈学习机制
5. 生产级监控与日志
"""

# 环境配置 ========================================================
import json
import logging
import time
import warnings
from typing import List, Dict

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
# 深度学习相关库
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    logging as hf_logging
)
from transformers.trainer_callback import EarlyStoppingCallback
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('companion_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 抑制不必要警告
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# 系统常量 ========================================================
class SystemConfig:
    """系统运行参数配置"""
    MAX_NEW_TOKENS = 128
    MODEL_PATH = "./DeepSeek-R1-Distill-Qwen-1.5B"
    DATA_FILE = "original_data.json"
    SAVE_PATH = "./production_model"
    DEVICE_MAP = {"": 0}  # 单GPU配置

    # 对话参数
    MAX_HISTORY = 3       # 最大对话轮次记忆
    TEMPERATURE = 0.85    # 生成多样性
    TOP_K = 50            # 采样参数
    TOP_P = 0.95          # 核心采样概率
    REPETITION_PENALTY = 1.5  # 重复抑制

    # 训练参数
    LORA_RANK = 16
    BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 2
    LEARNING_RATE = 5e-5

# 核心功能模块 ====================================================
class DataManager:
    """数据管理模块（支持热更新）"""

    @staticmethod
    def load_dataset(file_path: str) -> List[Dict]:
        """
        加载并验证训练数据集
        Args:
            file_path: 数据文件路径
        Returns:
            标准化后的数据集列表
        Raises:
            ValueError: 数据格式错误时抛出
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            # 强化数据校验
            if not isinstance(raw_data, list):
                raise ValueError("数据格式错误：顶层结构应为列表")

            validated_data = []
            required_keys = {'input', 'output'}
            for idx, item in enumerate(raw_data):
                if not isinstance(item, dict):
                    logger.warning(f"忽略第{idx + 1}条数据：非字典格式")
                    continue

                if not required_keys.issubset(item.keys()):
                    logger.warning(f"忽略第{idx + 1}条数据：缺少必要字段")
                    continue

                validated_data.append({
                    "input": str(item["input"]).strip(),
                    "output": str(item["output"]).strip(),
                    "emotion": str(item.get("emotion", "neutral")).lower()
                })

            if len(validated_data) == 0:
                raise ValueError("数据文件不包含有效数据")

            return validated_data

        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败：{str(e)}")
            raise
        except Exception as e:
            logger.error(f"数据加载异常：{str(e)}")
            raise


class ModelFactory:
    """模型生产工厂（集成量化与适配器）"""

    @classmethod
    def create_model(cls) -> tuple:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            SystemConfig.MODEL_PATH,
            pad_token='<|endoftext|>'
        )

        model = AutoModelForCausalLM.from_pretrained(
            SystemConfig.MODEL_PATH,
            quantization_config=quant_config,
            device_map=SystemConfig.DEVICE_MAP
        )

        lora_config = LoraConfig(
            r=SystemConfig.LORA_RANK,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            task_type="CAUSAL_LM"
        )
        return get_peft_model(model, lora_config), tokenizer

class TrainingEngine:
    """分布式训练引擎"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def _preprocess_data(self, dataset: List[Dict]) -> Dataset:
        def formatting_func(examples):
            inputs = [f"【对话】{ex['input']}" for ex in examples]
            model_inputs = self.tokenizer(
                inputs,
                max_length=256,
                truncation=True,
                padding='longest'
            )
            labels = self.tokenizer(
                [ex['output'] for ex in examples],
                max_length=256,
                truncation=True,
                padding='max_length'
            )
            model_inputs["labels"] = [
                [-100 if token == self.tokenizer.pad_token_id else token for token in seq]
                for seq in labels["input_ids"]
            ]
            return model_inputs

        return Dataset.from_list(dataset).map(formatting_func, batched=True)

    def execute_training(self, train_data: List[Dict], val_data: List[Dict]):
        train_dataset = self._preprocess_data(train_data)
        val_dataset = self._preprocess_data(val_data)

        training_args = TrainingArguments(
            output_dir='./prod_training',
            num_train_epochs=5,
            per_device_train_batch_size=SystemConfig.BATCH_SIZE,
            gradient_accumulation_steps=SystemConfig.GRAD_ACCUM_STEPS,
            learning_rate=SystemConfig.LEARNING_RATE,
            fp16=True,
            logging_dir='./logs',
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=200,
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.data_collator,
            callbacks=[EarlyStoppingCallback(patience=3)]
        )

        logger.info("启动模型训练...")
        trainer.train()
        self.model.save_pretrained(SystemConfig.SAVE_PATH)
        logger.info("模型保存完成")


class DialogueManager:

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []

    def _generate_response(self, prompt: str) -> str:
        """优化后的生成逻辑"""
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.model.device)

        with torch.inference_mode():
            # 中文生成强化
            generated_sequence = self.model.generate(
                **inputs,
                max_new_tokens=SystemConfig.MAX_NEW_TOKENS,
                temperature=SystemConfig.TEMPERATURE,
                top_p=SystemConfig.TOP_P,
                repetition_penalty=SystemConfig.REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])[0]  # 强制中文生成
            )

        # 解码和后处理
        response = self.tokenizer.decode(
            generated_sequence[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return self._postprocess_response(response)

    def _postprocess_response(self, text: str) -> str:
        """增强后处理"""
        # 过滤技术性表述
        filters = [
            "用户的回应：", "</think>", "首先", "其次",
            "assistant:", "user:", "response:", "answer:"
        ]
        for phrase in filters:
            text = text.replace(phrase, "")

        # 删除非中文字符
        text = ''.join([c for c in text if '\u4e00' <= c <= '\u9fff' or c in ['，', '。', '！', '？', '、']])

        # 截断到第一个标点
        for punc in ['。', '！', '？', '\n']:
            if punc in text:
                text = text.split(punc)[0] + punc
                break

        return text.strip()

    def process_message(self, user_input: str) -> str:
        self.history = [msg for msg in self.history if not msg.startswith("【")]
        self.history.append(f"用户：{user_input}")

        if len(self.history) > SystemConfig.MAX_HISTORY * 2:
            self.history = self.history[-SystemConfig.MAX_HISTORY*2:]

        context = "\n".join(self.history[-SystemConfig.MAX_HISTORY*2:])
        prompt = f"""根据对话历史用女友口吻回复：
{context}
请用1-2句话亲切回应：
"""
        response = self._generate_response(prompt)
        self.history.append(f"助手：{response}")
        return response

# 热更新模块 ======================================================
class ModelUpdater(FileSystemEventHandler):
    def __init__(self, training_engine):
        self.training_engine = training_engine
        self.last_modified = 0

    def on_modified(self, event):
        if event.src_path.endswith(SystemConfig.DATA_FILE):
            current_time = time.time()
            if current_time - self.last_modified > 5:
                self.last_modified = current_time
                logger.info("触发模型热更新...")
                self._perform_update()

    def _perform_update(self):
        """执行滚动更新"""
        try:
            new_data = DataManager.load_dataset(SystemConfig.DATA_FILE)
            if not new_data:
                logger.warning("没有有效的新数据，跳过更新")
                return

            # 确保数据分割正确
            val_size = max(1, int(len(new_data) * 0.1))
            train_data = new_data[:-val_size]
            val_data = new_data[-val_size:]

            logger.info(f"开始增量训练，数据量：{len(train_data)}训练/{len(val_data)}验证")

            # 执行训练
            self.training_engine.execute_training(train_data, val_data)
            logger.info("模型热更新完成")

        except Exception as e:
            logger.error(f"热更新失败：{str(e)}", exc_info=True)  # 记录完整堆栈信息

# 用户界面模块 ====================================================
class ChatInterface:
    @staticmethod
    def run_chat(model, tokenizer):
        dialogue_manager = DialogueManager(model, tokenizer)
        print("\n" + "="*40)
        print("  AI伴侣系统已启动（输入exit退出）")
        print("="*40)

        while True:
            try:
                user_input = input("你: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break

                start_time = time.time()
                response = dialogue_manager.process_message(user_input)
                latency = time.time() - start_time

                logger.info(f"生成耗时：{latency:.2f}s")
                print(f"AI: {response}")


            except KeyboardInterrupt:
                print("\n对话已终止")
                break

# 系统初始化 ======================================================
def initialize_system():
    logger.info("正在初始化系统...")
    model, tokenizer = ModelFactory.create_model()
    training_engine = TrainingEngine(model, tokenizer)

    observer = Observer()
    observer.schedule(ModelUpdater(training_engine), path='.', recursive=False)
    observer.start()

    return model, tokenizer, observer

# 主执行流程 ======================================================
if __name__ == "__main__":
    try:
        model, tokenizer, observer = initialize_system()
        ChatInterface.run_chat(model, tokenizer)
    finally:
        observer.stop()
        observer.join()
        logger.info("系统安全关闭")