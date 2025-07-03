import json, argparse, torch, os
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_DIR = "./doaug_artifacts/doaug_paraphraser"
HF_TOKEN = os.environ["HF_TOKEN"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K = 5
MAX_NEW_TOK = 64
SYSTEM_MSG = "You are a helpful assistant that only paraphrases."

tok = AutoTokenizer.from_pretrained(MODEL_DIR, token=HF_TOKEN)
END_ID = tok.convert_tokens_to_ids("<|eot_id|>")

PROMPT = "You will be given a sentence. Please paraphrase the sentence.\nSentence: "
# PROMPT = "Paraphrase the following sentence:\n"


def chat_prompt(s: str) -> str:
    chat = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"{PROMPT}{s}"},
    ]
    return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def generate_k(model, prompt: str, k: int):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    outs = model.generate(
        **inputs,
        num_beams=k,
        num_return_sequences=k,
        max_new_tokens=MAX_NEW_TOK,
        early_stopping=True,
        eos_token_id=END_ID,
    )
    pl = inputs["input_ids"].shape[1]
    res = []
    for o in outs:
        txt = tok.decode(o[pl:], skip_special_tokens=False)
        res.append(txt.split("<|eot_id|>")[0].strip())
    return res


def most_distant(paraphrases, original, embedder):
    embs = embedder.encode(
        [original] + paraphrases,
        convert_to_tensor=True,
        device=DEVICE,
        normalize_embeddings=True,
    )
    base = embs[0]
    dists = 1 - (embs[1:] @ base)
    return paraphrases[int(torch.argmax(dists))]


def main(inp: Path, out: Path):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16, token=HF_TOKEN
    )
    model.eval()
    embedder = SentenceTransformer(EMB_MODEL, device=DEVICE)

    ds = load_dataset("json", data_files=str(inp))["train"]
    with out.open("w", encoding="utf-8") as fout:
        for ex in tqdm(ds, desc="augmenting"):
            sent = ex["sentence"]
            cands = generate_k(model, chat_prompt(sent), K)
            best = most_distant(cands, sent, embedder)
            fout.write(
                json.dumps({"sentence": sent, "paraphrase": best}, ensure_ascii=False)
                + "\n"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="./test_samples/input.jsonl")
    p.add_argument("--output", default="./test_samples/output.jsonl")
    args = p.parse_args()
    main(Path(args.input), Path(args.output))
