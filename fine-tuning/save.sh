#!/bin/sh
#SBATCH --time=3:00:00
#SBATCH --job-name=mytestjob
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@comp.nus.edu.sg
#SBATCH --gpus=1

#which python
#source /home/j/jiayuan1/.jy/jy_venv/bin/activate
#python train.py
python save.py --local_model_name "summary_anomaly_llm" --hub_model_name "jiayuan1/summary_anomaly_llm" --quantization 0 #save_local
#python llama.cpp/convert_hf_to_gguf.py bamboo-llm \
#	--outfile bamboo-llm.gguf \
#	--outtype q8_0
