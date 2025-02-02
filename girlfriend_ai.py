from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
from bitsandbytes.nn import Int8Params
import json
import warnings
import logging
from transformers import logging as hf_logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transformers.trainer_callback import EarlyStoppingCallback

# # 抑制警告信息
# warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
# warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.")
# # 调整日志等级，只显示错误信息
# logging.basicConfig(level=logging.ERROR)
# hf_logging.set_verbosity_error()

# 定义女友前缀
girlfriend_prefix = "你"

# 从外置文件读取数据
def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                line_num = e.lineno
                start = max(0, line_num - 5)
                end = line_num + 5
                lines = content.splitlines()
                print(f"错误发生在第 {line_num} 行，附近内容如下：")
                for i in range(start, end):
                    if i < len(lines):
                        print(f"{i + 1}: {lines[i]}")
                raise
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查文件路径。")
        return []

# 数据处理
def preprocess_function(examples, tokenizer):
    inputs = [girlfriend_prefix + "用女友口吻回复：" + input_text for input_text in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(examples["output"], max_length=128, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    labels_mask = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in label] for label in model_inputs["labels"]]
    model_inputs["labels"] = labels_mask
    return model_inputs


def train_and_save_model(model, training_args, train_dataset, eval_dataset, data_collator, save_path):
    try:
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=[early_stopping_callback]
        )
        total_steps = len(train_dataset) // (
                    training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs
        trainer.create_optimizer_and_scheduler(num_training_steps=total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=total_steps
        )
        trainer.lr_scheduler = scheduler
        trainer.train()
        torch.cuda.empty_cache()
        model.save_pretrained(save_path)
        return model
    except Exception as e:
        print(f"Training error: {e}")
        return model


class MyHandler(FileSystemEventHandler):
    def __init__(self, model, training_args, tokenizer, data_collator, save_path):
        self.model = model
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.save_path = save_path
        self.last_modified = time.time()

    def on_modified(self, event):
        if event.src_path.endswith('original_data.json'):
            current_modified = time.time()
            if current_modified - self.last_modified > 1:
                self.last_modified = current_modified
                print(f"Detected change in {event.src_path}")
                self.process_new_data()

    def process_new_data(self):
        try:
            new_data = load_data('original_data.json')
            new_dataset = Dataset.from_list(new_data)
            new_tokenized_dataset = new_dataset.map(
                preprocess_function,
                batched=True,
                fn_kwargs={"tokenizer": self.tokenizer}
            )
            # 划分训练集和验证集
            split = new_tokenized_dataset.train_test_split(test_size=0.1)
            train_data = split["train"]
            eval_data = split["test"]
            self.model = train_and_save_model(
                model=self.model,
                training_args=self.training_args,
                train_dataset=train_data,
                eval_dataset=eval_data,
                data_collator=self.data_collator,
                save_path=self.save_path
            )
            print("Model fine-tuned with new data.")
        except Exception as e:
            print(f"Error processing data: {e}")


if __name__ == "__main__":
    # 从本地加载分词器
    tokenizer = AutoTokenizer.from_pretrained("./DeepSeek-R1-Distill-Qwen-1.5B")
    # 指定 eos_token 作为填充标记
    tokenizer.pad_token = tokenizer.eos_token

    # 自定义 device_map
    device_map = {
        '': 0  # 将整个模型加载到 GPU 0 上，如果有多个 GPU，可以根据需要调整
    }

    # 配置量化参数
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )

    # 加载 8 位量化模型并启用 CPU 卸载
    model = AutoModelForCausalLM.from_pretrained(
        "./DeepSeek-R1-Distill-Qwen-1.5B",
        quantization_config=quantization_config,
        device_map=device_map
    )

    # 配置 LoRA 适配器
    config = LoraConfig(
        r=16,  # LoRA 秩
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # 要应用 LoRA 的模块
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 将 LoRA 适配器应用到模型上
    model = get_peft_model(model, config)

    # 定义提前停止回调，当验证损失在 3 个 epoch 内没有改善时停止训练
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=20,  # 训练轮数
        per_device_train_batch_size=4,  # 批次大小
        gradient_accumulation_steps=2,  # 梯度累积步数
        warmup_steps=200,  # 预热步数
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        learning_rate=5e-5,  # 学习率
        lr_scheduler_type="cosine",  # 使用线性退火学习率调度
        report_to="tensorboard",  # 数据可视化，目前没用

        load_best_model_at_end=True,
        metric_for_best_model="loss",
        eval_strategy="steps",  # 修改为 eval_strategy
        save_strategy="steps",  # 添加或修改保存策略为 steps
        eval_steps=50  # 添加评估步数，可按需调整
    )


    # 自定义数据收集器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # 加载原始数据
    data = load_data('original_data.json')

    # 将数据转换为数据集对象
    dataset = Dataset.from_list(data)

    # 划分训练集和验证集
    train_dataset = dataset.train_test_split(test_size=0.1)["train"]
    eval_dataset = dataset.train_test_split(test_size=0.1)["test"]

    # 对训练集和验证集应用预处理函数
    cache_file_train = "./tokenized_dataset_cache_train"
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=12,
                         cache_file_name=cache_file_train, fn_kwargs={"tokenizer": tokenizer})

    cache_file_eval = "./tokenized_dataset_cache_eval"
    tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=12,
                        cache_file_name=cache_file_eval, fn_kwargs={"tokenizer": tokenizer})


    # 开始训练
    model = train_and_save_model(model, training_args, tokenized_train_dataset, tokenized_eval_dataset, data_collator,"./fine_tuned_girlfriend_ai")
    # 创建事件处理对象
    event_handler = MyHandler(model, training_args, tokenizer, data_collator, "./fine_tuned_girlfriend_ai")
    # 创建观察者对象
    observer = Observer()
    # 监控当前目录下的文件变化
    observer.schedule(event_handler, path='.', recursive=False)
    # 启动观察者
    observer.start()

    device = next(model.parameters()).device  # 获取模型所在设备

    try:
        while True:
            user_input = input("你: ").strip()
            if not user_input:
                continue
            input_text = girlfriend_prefix + "用可爱女友口吻简短回复：" + user_input
            encoding = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            # 截断输入部分，仅保留生成内容
            response_ids = output[0][len(input_ids[0]):]
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            print(" AI:", response)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()