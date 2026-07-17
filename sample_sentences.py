import sys, argparse, json, os, tomllib
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))


def sample_from_model(model_cfg, n, device, hf_cache_dir, batch_size, temperature, max_new_tokens):
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], cache_dir=hf_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], dtype=torch.bfloat16, cache_dir=hf_cache_dir
    ).to(device)
    model.eval()

    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    sentences = []

    with torch.no_grad():
        for _ in tqdm(range(0, n, batch_size), desc=f"Sampling {model_cfg['name']}"):
            batch = min(batch_size, n - len(sentences))
            input_ids = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
            for out in outputs:
                text = tokenizer.decode(out[1:], skip_special_tokens=True).strip()
                if text:
                    sentences.append(text)

    del model
    torch.cuda.empty_cache()
    return sentences[:n]


def main():
    parser = argparse.ArgumentParser(description="Sample sentences from source and target models")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--n", type=int, default=1000, help="Sentences to sample per model")
    parser.add_argument("--output", default="data/generated_sentences.json")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    hf_cache_dir = os.getenv("HF_CACHE_DIR")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config.get("batch_size", 32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.n} sentences from {config['source_model']['name']}...")
    source_sentences = sample_from_model(
        config["source_model"], args.n, device, hf_cache_dir, batch_size, args.temperature, args.max_new_tokens
    )

    print(f"Sampling {args.n} sentences from {config['target_model']['name']}...")
    target_sentences = sample_from_model(
        config["target_model"], args.n, device, hf_cache_dir, batch_size, args.temperature, args.max_new_tokens
    )

    all_sentences = list(dict.fromkeys(source_sentences + target_sentences))
    with open(output_path, "w") as f:
        json.dump(all_sentences, f, indent=2)

    print(f"Saved {len(all_sentences)} unique sentences to {output_path}")


if __name__ == "__main__":
    main()
