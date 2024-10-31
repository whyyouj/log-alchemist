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
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

max_seq_length = 8192
dtype = None
load_in_4bit = True
alpaca_prompt = """
{}
### QUERY:
{}

Variable `dfs: list[pd.DataFrame]` is already declared.

At the end, declare "result" variable as a dictionary of type and value.

If you are asked to plot a chart, use "matplotlib" for charts, save as png.


Generate python code and return full updated code:

You are required to answer the QUERY"""


INSTRUCTION_0 = """
You are a data analyst that has been tasked with the goal of  coming up with python codes.

<dataframe>
dfs[0]:2000x9
LineId,User,Component,PID,Address,Content,EventId,EventTemplate,Datetime
1683,calvisitor-10-105-162-228,networkd,33847,[32626],Cocoa scripting error for '0x0067000f': four character codes must be four characters long.,E275,<*>-<*>-<*> <*>:<*>:<*>.<*> ksfetch[<*>/<*>] [lvl=<*>] main() ksfetch fetching URL (<NSMutableURLRequest: <*>> { URL: <*> }) to folder:<*>,2024-07-03 23:29:01
484,calvisitor-10-105-163-147,com.apple.xpc.launchd,32776,com.apple.xpc.launchd.domain.pid.WebContent.37963,2017-07-02 15:46:41.445 ksfetch[32435/0x7fff79824000] [lvl=2] main() ksfetch fetching URL (<NSMutableURLRequest: 0x1005110b0> { URL: https://tools.google.com/service/update2?cup2hreq=53f725cf03f511fab16f19e789ce64aa1eed72395fc246e9f1100748325002f4&cup2key=7:1132320327 }) to folder:/tmp/KSOutOfProcessFetcher.YH2CjY1tnx/download,E229,"NSURLSession/NSURLConnection HTTP load failed (kCFStreamErrorDomainSSL, <*>)",2024-07-04 07:54:21
102,calvisitor-10-105-160-22,locationd,37682,,Sandbox: com.apple.Addres(34685) deny(1) network-outbound /private/var/run/mDNSResponder,E222,[HID] [MT] AppleMultitouchDevice::willTerminate entered,2024-07-04 07:40:52
</dataframe>

You are already provided with the following functions that you can call:
<function>
def overall_summary(df):

    Use this for any question regarding an Overall Summary
    The output type will be a string
    Args:
        df pd.DataFrame: A pandas dataframe

</function>
<function>
def overall_anomaly(df):

    Use this for any question regarding an Overall Anomaly 	analysis of the data set
    The output type will be a string
    Args:
        df pd.DataFrame: A pandas dataframe

</function>
Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd
# Write code here
# Declare result var:
type (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }
```
"""
INSTRUCTION_1="""
You are a data analyst that has been tasked with the goal of  coming up with python codes.

<dataframe>
dfs[0]:536350x8
TransactionNo,Date,ProductNo,ProductName,Price,Quantity,CustomerNo,Country
553012,8/30/2019,21809,Hanging Metal Rabbit Decoration,11.26,-60,,Lithuania
548511,7/17/2019,85136B,Chest Of Drawers Gingham Heart,90.5,164,15837.0,European Community
550960,7/26/2019,21813,Pink Combo Mini Crystals Necklace,25.16,188,12450.0,Malta
</dataframe>

You are already provided with the following functions that you can call:
<function>
def overall_summary(df):

    Use this for any question regarding an Overall Summary
    The output type will be a string
    Args:
        df pd.DataFrame: A pandas dataframe

</function>
<function>
def overall_anomaly(df):

    Use this for any question regarding an Overall Anomaly 	analysis of the data set
    The output type will be a string
    Args:
        df pd.DataFrame: A pandas dataframe

</function>
Update this initial code:
```python
# TODO: import the required dependencies
import pandas as pd
# Write code here
# Declare result var:
type (possible values "string", "number", "dataframe", "plot"). Examples: { "type": "string", "value": f"The highest salary is {highest_salary}." } or { "type": "number", "value": 125 } or { "type": "dataframe", "value": pd.DataFrame({...}) } or { "type": "plot", "value": "temp_chart.png" }
```
"""


input = "filter the data and give me the mac users"
huggingface_model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

#huggingface_model_name = "meta-llama/Llama-3.1-8B-Instruct"
#huggingface_model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
huggingface_model_name = "NousResearch/Hermes-3-Llama-3.1-8B" #"NousResearch/Meta-Llama-3.1-8B"# "router_llm"
# 2. Before Training

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = huggingface_model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = HF
)



# 3. Load data
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)
#EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instruction = examples["Value"]
    input       = examples["Input"]
    output      = examples["Output"]
    texts = []

    if instruction == 0:
        instruction = INSTRUCTION_0
    else:
        instruction = INSTRUCTION_1
    input = {"role":"user", "content" :alpaca_prompt.format(instruction, input)}
    output =  {"role": "assistant", "content": output}
    
    texts = tokenizer.apply_chat_template([input, output], tokenize = False, add_generation_prompt = False)
    
    return { "text" : texts, }

dataset = load_dataset("csv", data_files="data/train_python.csv", split = "train")

dataset = dataset.map(formatting_prompts_func, batched = False,)
print(dataset[5]["text"])

# 4. Training
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],#,"embed_tokens", "lm_head"],
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
    #data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #num_train_epochs = 200, # Set this for 1 full training run
        max_steps = 20000,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "python",
    ),
)


gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
'''from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user<|im_end|>\n\n",
    response_part = "<|im_start|>assistant<|im_end|>\n\n",
)'''

trainer_stats = trainer.train()



FastLanguageModel.for_inference(model)
inputs = tokenizer(
        [
            alpaca_prompt.format(
                INSTRUCTION_0,
                input,
                ""
            )
        ], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)
model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1024)
model.save_pretrained("python_llm_v3.1")
tokenizer.save_pretrained("python_llm_v3.1")
huggingface_model_name = "jiayuan1/python_llm_v3"

model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "q4_k_m", token = HF)
model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = "q5_k_m", token = HF)
