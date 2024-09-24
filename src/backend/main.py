from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

#set config for model quantisation
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "microsoft/Phi-3-small-128k-instruct"
tokenizer_id = "microsoft/Phi-3-small-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='auto', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    #device_map={"": accelerator.process_index}
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/run_slm")
async def get_slm_response(prmpt: str, messages: list) -> dict:
    output = pipe(messages, **generation_args)
    res = output[0]['generated_text']
    response = {"res": ""}
    response["res"] = res
    print('response: ', response)
    return response