import argparse
import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from acttrans.utils.config import load_config
from acttrans.utils.hf import load_model_and_tokenizer


def sample_from_model(model_cfg, n, device, batch_size, temperature, max_new_tokens):
    model, tokenizer = load_model_and_tokenizer(model_cfg["name"], device)
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

    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config.get("batch_size", 32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.n} sentences from {config['source_model']['name']}...")
    source_sentences = sample_from_model(
        config["source_model"], args.n, device, batch_size, args.temperature, args.max_new_tokens
    )

    print(f"Sampling {args.n} sentences from {config['target_model']['name']}...")
    target_sentences = sample_from_model(
        config["target_model"], args.n, device, batch_size, args.temperature, args.max_new_tokens
    )

    all_sentences = list(dict.fromkeys(source_sentences + target_sentences))
    with open(output_path, "w") as f:
        json.dump(all_sentences, f, indent=2)

    print(f"Saved {len(all_sentences)} unique sentences to {output_path}")


if __name__ == "__main__":
    main()
