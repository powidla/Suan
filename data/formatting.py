from datasets import load_dataset
from data import prompt_formats


class FormatAlpaca:
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("tatsu-lab/alpaca", split=self.split)
        self.data = self.data.filter(lambda x: x["output"].strip() != "")
        self.data = self.data.filter(lambda x: x["instruction"].strip() != "")

    @property
    def sft_format(self):
        data = self.data.map(lambda x: {
        	"prompt": prompt_formats.PROMPT.format(x["instruction"].strip(), x["input"].strip()),
        	"completion": prompt_formats.ANSWER.format(x["output"].strip()) + self.tokenizer.eos_token},
        remove_columns=["instruction", "input", "output", "text"])
        return data


class FormatXSTest:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/XS-Test/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")
        self.data = self.data.filter(lambda x: x["label"] == "safe")

    @property
    def sft_format(self):
        data = self.data.map(lambda x: {
        	"prompt": prompt_formats.build_prompt(x["prompt"], use_sys_prompt=self.use_sys_prompt),
			"label": x["label"]},
        remove_columns=["prompt", "focus", "type", "note", "label"])
        return data


class FormatMaliciousInstruct:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/MaliciousInstruct/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        return self.data.map(lambda x: {"prompt": prompt_formats.build_prompt(x["prompt"], use_sys_prompt=self.use_sys_prompt)})


class FormatAdvBench:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/AdvBench/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        return self.data.map(lambda x: {"prompt": prompt_formats.build_prompt(x["prompt"], use_sys_prompt=self.use_sys_prompt)})


class FormatHarmBench:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/HarmBench/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        return self.data.map(lambda x: {"prompt": prompt_formats.build_prompt(x["prompt"], use_sys_prompt=self.use_sys_prompt)})


class FormatHHRLHF:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("parquet", data_dir="sets/HH-RLHF/data", split=self.split)

    @property
    def dpo_format(self):
        data = self.data.map(lambda x: {
            "prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), ""),
            "chosen": prompt_formats.ANSWER.format(x["chosen"].strip()),
            "rejected": prompt_formats.ANSWER.format(x["rejected"].strip())})
        return data


class FormatPKUDPO:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("parquet", data_dir="sets/PKU/data", split=self.split)

    @property
    def dpo_format(self):
        data = self.data.map(lambda x: {
            "prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), ""),
            "chosen": prompt_formats.ANSWER.format(x["chosen"].strip()),
            "rejected": prompt_formats.ANSWER.format(x["rejected"].strip())})
        return data


class FormatPKUSafe:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("PKU-Alignment/PKU-SafeRLHF-30K", split=self.split)

    @property
    def dpo_format(self):
        data = self.data.map(lambda x: {
                "prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), ""),
                "chosen": prompt_formats.ANSWER.format(x["response_0"].strip()),
                "rejected": prompt_formats.ANSWER.format(x["response_1"].strip()),
                "chosen_is_safe": x["is_response_0_safe"],
                "rejected_is_safe": x["is_response_1_safe"]},
            remove_columns=["response_0", "response_1", "is_response_0_safe", "is_response_1_safe", "better_response_id", "safer_response_id"])
        data = data.map(self.swap)
        data = data.filter(lambda x: x["chosen"] is not None)
        return data

    @staticmethod
    def swap(x):
        h_plus = not x["chosen_is_safe"]
        h_minus = not x["rejected_is_safe"]

        if not h_plus:
            return x
        elif not h_minus:
            return {**x,
                    "chosen": x["rejected"],
                    "rejected": x["chosen"],
                    "chosen_is_safe": x["rejected_is_safe"],
                    "rejected_is_safe": x["chosen_is_safe"]}
        else:
            return {**x, "chosen": None, "rejected": None}


class FormatSorryBench:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/SorryBench/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        return self.data.map(lambda x: {"prompt": prompt_formats.build_prompt(x["prompt"], use_sys_prompt=self.use_sys_prompt)})


class FormatORBench:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/OR-Bench/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        return self.data.map(lambda x: {"prompt": prompt_formats.build_prompt(x["prompt"], use_sys_prompt=self.use_sys_prompt)})


class FormatMTBench:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/MT-Bench/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        return self.data.map(lambda x: {"prompt": prompt_formats.build_prompt(x["prompt"], use_sys_prompt=self.use_sys_prompt)})


class FormatAlpacaEval:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/AlpacaEval/data", split=self.split)
        self.data = self.data.filter(lambda x: x["instruction"].strip() != "")

    @property
    def sft_format(self):		
        use_sys_prompt = self.use_sys_prompt
        data = self.data.map(lambda x: {"prompt": prompt_formats.PROMPT.format(x["instruction"].strip(), "")})
        return data


class FormatArenaHard:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("parquet", data_dir="sets/ArenaHard/data", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):		
        use_sys_prompt = self.use_sys_prompt
        data = self.data.map(lambda x: {"prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), "")}, remove_columns=["prompt", "uid", "category", "cluster"])
        return data


class FormatNoveltyBench:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("yimingzhang/novelty-bench", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        use_sys_prompt = self.use_sys_prompt
        data = self.data.map(lambda x: {
        	"prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), "")},
        remove_columns=["id"])
        return data


class FormatARC:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=self.split)
    
    @property
    def sft_format(self):
        use_sys_prompt = self.use_sys_prompt
        data = self.data.map(lambda x: {
        	"prompt": prompt_formats.PROMPT.format(prompt_formats.build_mmlu_prompt(x["question"].strip(), x["choices"]["text"], x["choices"]["label"]), ""),
        	"correct_response": x["answerKey"]},
        remove_columns=["id", "choices", "answerKey", "question"])
        return data


class FormatMMLU:
    def __init__(self, split, use_sys_prompt=False):
        self.split = split
        self.use_sys_prompt = use_sys_prompt
        self.data = load_dataset("cais/mmlu", "all", split=self.split)
    
    @property
    def sft_format(self):
        use_sys_prompt = self.use_sys_prompt
        data = self.data.map(lambda x: {
        	"prompt": prompt_formats.PROMPT.format(prompt_formats.build_mmlu_prompt(x["question"].strip(), x["choices"]), ""),
        	"correct_response": "ABCD"[x["answer"]]},
        remove_columns=["subject", "choices", "answer", "question"])
        return data


class FormatEval:
    def __init__(self, data):
        self.data = data

    def clean_instruction(self, prompt):
        if "<|instruct|>" in prompt and "<|/instruct|>" in prompt:
            instruction = prompt.split("<|instruct|>")[1].split("<|/instruct|>")[0].strip()
            return instruction
        return prompt

    def clean_response(self, response):
        if "<|response|>" in response:
            response = response.split("<|response|>")[1]
        if "<|/response|>" in response:
            response = response.split("<|/response|>")[0]
        return response.strip()
		
    @property
    def judge_arena_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.DEFAULT_EVAL_PROMPT.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
    
    @property
    def judge_flow_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.FLOW_JUDGE_PROMPT.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
    
    @property
    def judge_refuse_flow_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.FLOW_JUDGE_PROMPT_OVERREFUSE.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
	
    @property
    def judge_instruct_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_INSTRUCT.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
    
    @property
    def judge_safe_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_SAFETY.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
		
    @property
    def judge_refusal_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_REFUSE.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
