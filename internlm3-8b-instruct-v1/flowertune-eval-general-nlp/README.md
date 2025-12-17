# General NLP Evaluation Guide (InternLM3-8B-Instruct)

We evaluate on MMLU (STEM, Social Sciences, Humanities) following the Flower leaderboard rules.

## Environment setup
- Install deps: `pip install -r requirements.txt`
- Hugging Face auth: `huggingface-cli login`

## Run example
- Default: 4bit quantization is required for leaderboard (`--quantization=4`), batch size 16.
- Base model: `internlm/internlm3-8b-instruct`
- PEFT checkpoint: `./workspace/results/<timestamp>/peft_10`

```bash
python eval.py --base-model-name-path=internlm/internlm3-8b-instruct --peft-path=./workspace/results/<timestamp>/peft_10 --run-name=eval_internlm3 --batch-size=16 --quantization=4 --category=stem,social_sciences,humanities
```

## Outputs
- Generations/accuracy: `benchmarks/generation_{dataset}_{category}_{run_name}.jsonl`, `benchmarks/acc_{dataset}_{category}_{run_name}.txt`
- No extra public benchmarks beyond `internlm3-8b-instruct-v1/flowertune-eval-general-nlp/benchmarks`
