import argparse
import os
import json
import re

from tqdm import tqdm
from huggingface_hub import hf_hub_download, upload_file


def extract_ans(text):
    m = re.search(r'Answer:\s*([A-Z]|\d+)', text)
    return m.group(1) if m else None

class Evaluator:
    def __init__(self, input_file, output_dir, repo_id):

        self.input_file = input_file
        self.repo_id = repo_id
        self.model_id = f"{self.input_file.split('/')[-2]}"

        self.output_dir = os.path.join(output_dir, self.model_id)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading data from: {self.input_file}")
        self.data = self.load_file(self.input_file)

    def load_file(self, path):

        if not os.path.isdir(path):
            print(f"Downloading from {path} ...")

            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=path,
                repo_type="dataset"
            )
            print(f"Downloaded to: {path}")

        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

        return data

    def compute_winrate(self, model_response, gold):
        
        pred = extract_ans(model_response)

        return int(pred == gold)

    def run(self):

        total = 0
        correct = 0

        for item in tqdm(self.data, desc="Computing win rate"):
            gold = item["correct_response"].strip()
            model_resp = item["responses"][0]

            total += 1
            correct += self.compute_winrate(model_resp, gold)

        wr = correct / total if total > 0 else 0.0

        score_path = os.path.join(self.output_dir, "score.jsonl")
        with open(score_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"wr": wr}, ensure_ascii=False) + "\n")

        if self.repo_id is not None:
            upload_file(path_or_fileobj=score_path, path_in_repo=f"{self.model_id}/score.jsonl",repo_id=self.repo_id, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description="Compute win rate for ARC/MMLU")
    parser.add_argument("--input_file", type=str, required=True, help='Path to JSONL file with prompts and responses')
    parser.add_argument("--output_dir", type=str, default="./outputs", help='Directory to save scored responses')
    parser.add_argument("--repo_id", type=str, required=True, help="HF repository to push to")

    args = parser.parse_args()

    evaluator = Evaluator(
        input_file=args.input_file,
        output_dir=args.output_dir,
        repo_id=args.repo_id
    )

    evaluator.run()


if __name__ == "__main__":
    main()
