import os
from dotenv import load_dotenv

load_dotenv()
HF = os.getenv("HF_TOKEN")

from unsloth import FastLanguageModel
import torch
import os
from transformers import TextStreamer
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


max_seq_length = 2048
dtype = None
load_in_4bit = True
alpaca_prompt = """
### Instruction:
Please extract all individual actions from the following input and categorize them into one of three categories: 
Pandas (for dataframe-related actions), Explain (for explanation requests), or General (for non-dataframe related questions).
- Group dependent actions into the same sentence. 
- Split explanations into separate entries.
- General questions (non-dataframe related) should also be in separate entries.

Return each categorized action in a list format, where each entry is a dictionary containing the category and the associated action.


### Input:
{}

### Response:
{}"""


#instruction = "What category does this question fall under: Summary, Anomaly, General"
input = "plot the graph of users and macbook and what is usa"

#huggingface_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
#huggingface_model_name = "meta-llama/Llama-3.1-8B-Instruct"
huggingface_model_name = "NousResearch/Meta-Llama-3-8B-Instruct"

# 2. Before Training
'''
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF
)

'''
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = huggingface_model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF
)



# 3. Load data

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):

    input       = examples["Input"]
    output      = examples["Output"]
    texts = []
    texts = alpaca_prompt.format(input, output) + EOS_TOKEN
    return { "text" : texts, }

dataset = load_dataset("csv", data_files="data/train_router_5.csv", split = "train")

dataset = dataset.map(formatting_prompts_func, batched = False,)

# 4. Training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True, #"unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 50, # Set this for 1 full training run.
        #max_steps = 13000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "router",
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

#trainer_stats = trainer.train()

trainer_stats = trainer.train(resume_from_checkpoint=True)

'''used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")'''


FastLanguageModel.for_inference(model)
inputs = tokenizer(
        [
            alpaca_prompt.format(
                input,
                ""
            )
        ], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)
model.save_pretrained("router_llm")
tokenizer.save_pretrained("router_llm")
model.save_pretrained_gguf("router_llm_gguf")
huggingface_model_name = "jiayuan1/router_llm"
model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "q4_k_m", token = HF)
model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "q5_k_m", token = HF)
