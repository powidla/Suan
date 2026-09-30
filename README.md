# Suan

## Installation
Run the following to create a conda environment with the necessary dependencies.
```bash
conda create -n siu python=3.11
```
Next, after the activation of ```siu``` environment, install required libraries.
```bash
pip install einops
pip install peft
pip install trl
pip install bitsandbytes
pip install tiktoken
pip install pydantic
pip install vllm 
```

Next, after the activation of ```siu``` environment, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

### Quantization
You can quantize a model using the following command:
```bash
python utils/quantize.py --model /path/to/model --output_dir /path/to/output/directory
```

### Supervised Fine-Tuning and DPO Stage 
To fine-tune quantized model on Alpaca dataset use the following command:
```bash
python train/train_sft.py --model /path/to/model --dataset alpaca --loss ce --output_dir /path/to/output/directory --max_length $MAX_LENGTH
```
Further, to perform preference optimization run this command:
```bash
python train/train_q_dpo.py --dataset $DATASET --loss $LOSS --model /path/to/model --peft_weights /path/to/peft --output_dir /path/to/output/directory --max_length $MAX_LENGTH
```
Here $\text{DATASET}$ supports PKU-Safe-RLHF dataset denoted ```pku``` and HH-RLHF dataset denoted ```hh```,  while $\text{LOSS}$ accepts three options, DPO ```dpo```, IPO ```ipo``` and our loss Suan ```siu```.
To train Safe-DPO run:
```bash
python train/train_q_safedpo.py --dataset pku-safe --loss safe-dpo --model /path/to/model --peft_weights /path/to/peft --output_dir /path/to/output/directory --max_length $MAX_LENGTH
```

### Inference
To run the inference of Base version of the model execute the following command:
```bash
python inference/inference_sft.py --base_model /path/to/base_model --dataset $DATASET --dataset_format sft --quantization bitsandbytes
```
Supported datasets: Malicious Instruct - ```mi```, HarmBench - ```hb```, AdvBench - ```ab```, SORRY-Bench - ```sb```, XS-Test - ```xs```, OR-Bench - ```or```, AlpacaFarm - ```ae```, MT-Bench - ```mt```, Arena Hard - ```ah```, ARC - ```arc```, MMLU - ```mmlu```, Novelty Bench - ```nb```. 

Similarly, the inference of SFT/DPO model requires running the following command:
```bash
python inference/inference_sft.py --base_model /path/to/base_model --peft_weights /path/to/peft --dataset $DATASET --dataset_format sft --quantization bitsandbytes
```
Where peft weights accept both SFT and DPO models stored in the output directory. 

### Evaluation
For safety evaluation on ```mi, hb, ab, sb``` execute this command:
```bash
python evaluation/eval_asr.py --input_file /path/to/responses --output_dir path/to/results 
```
To evaluate the compliance of the model on ```xs, or``` run the following:
```bash
python evaluation/eval_refusal.py --input_file /path/to/responses --output_dir path/to/results 
```
Accessing model helpfulness on ```ae, mt, ah``` can be achieved by using the code below:
```bash
python evaluation/evaluation_judge.py --judge_model "flowaicom/Flow-Judge-v0.1" --input_file /path/to/responses --output_dir path/to/results --max_new_tokens $MAX_LENGTH
```
Calculating the win rate on ARC and MMLU is done via:
```bash
python evaluation/evaluation_factuality.py --input_file /path/to/responses --output_dir path/to/results
```
Finally, to compare the performance on NoveltyBench we refer to the original [repo](https://github.com/novelty-bench/novelty-bench). 

As our code for the inference and evaluation relies on vLLM engine, one might consider using ```export VLLM_USE_FLASHINFER_SAMPLER=0``` commands to avoid errors with ```python 3.11```. 

### Citation 
If you find this repo useful, please consider citing our work
```
@misc{cherednichenko2026suanrectifyingdirectpreference,
      title={Suan: Rectifying Direct Preference Safety Alignment in Large Language Models}, 
      author={Oleksandr Cherednichenko and Roman Klypa},
      year={2026},
      eprint={2609.08634},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.08634}, 
}
```
### License
The code is released under the Apache 2.0 license. 
