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
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


instruction = "What category does this question fall under: Summary, Anomaly, General"
input = "how many outliers are there in the data"
huggingface_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

#huggingface_model_name = "meta-llama/Llama-3.1-8B-Instruct"
#huggingface_model_name = "NousResearch/Meta-Llama-3-8B-Instruct"
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

'''

FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        instruction, # instruction
        input, # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)
#_ = tokenizer.decode(output_ids[0], skip_special_tokens=True)
'''
# Print the output
#print("Generated Output:")
#print(output_text)


# 3. Load data

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instruction = "Which category does this question fall under: Summary, Anomaly, General" #examples["instruction"]
    input       = examples["Input"]
    output      = examples["Output"]
    texts = []
    #similar_dict = {}
    #if not similar_dict.get(input, False):
    #    similar_dict[input] = True
    #for instruction, input, output in zip(instructions, inputs, outputs):
    texts = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
    #    texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("jiayuan1/summary_anomaly_dataset", split = "train")

#dataset = dataset.map(formatting_prompts_func, batched = False,)
df = dataset.to_pandas()
df_unique = df.drop_duplicates()
dataset_unique = Dataset.from_pandas(df_unique)
dataset = dataset_unique.map(formatting_prompts_func, batched = False)
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
        num_train_epochs = 100, # Set this for 1 full training run.
        #max_steps = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "summary_anomaly_v2",
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#trainer_stats = trainer.train(resume_from_checkpoint=True)

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
                instruction,
                input,
                ""
            )
        ], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000)
model.save_pretrained("summary_anomaly_llm_v2")
tokenizer.save_pretrained("summary_anomaly_llm_v2")
#model.push_to_hub(huggingface_model_name, token = HF) 
#tokenizer.push_to_hub(huggingface_model_name, token = HF)
#model.save_pretrained_gguf('summary_anomaly_model', tokenizer, quantization_method = "")
#huggingface_model_name = "jiayuan1/Llama-3.1_test"
#model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "f16", token = HF)
