import argparse
from dotenv import load_dotenv
import os

load_dotenv()
HF = os.getenv("HF_TOKEN")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_model_name", type=str,required=True)
    parser.add_argument("--save_local", action='store_true')
    parser.add_argument("--hub_model_name", type = str, required = False)
    parser.add_argument("--quantization", type =int, default = 0)


    args = parser.parse_args()
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.local_model_name,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    quantization_method = ['q4_k_m', 'q8_0', 'q5_k_m', 'f16'][args.quantization]
    if args.save_local:
        model.save_pretrained_gguf(args.local_model_name+'_gguf', tokenizer, quantization_method = quantization_method)
        print("local")
    if args.hub_model_name: 
        huggingface_model_name = args.hub_model_name
        model.push_to_hub_gguf(huggingface_model_name, tokenizer, quantization_method = [quantization_method], token = HF)
        print(quantization_method)
    return

main()
