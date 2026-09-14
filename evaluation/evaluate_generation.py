"""
Generation Evaluation for Monkey-Paw-RAG

Metrics:
- BLEU
- ROUGE-L
- METEOR
- BERTScore
"""

import json
import sys
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score

# ============================================================
# Config
# ============================================================

K=5
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from retrieve import Retriever
from chat import generate_answer
DATASET_PATH = ROOT_DIR / "evaluation" / "datasets" / "dataset_ground_truth.json"
OUTPUT_PATH = ROOT_DIR / "evaluation" / "results"/ "generation_evaluation.json"

# ============================================================
# Metrics
# ============================================================

def bleu(reference, generated):
    return sentence_bleu(
        [reference.split()],
        generated.split(),
        smoothing_function=SmoothingFunction().method1
    )

def rouge_l(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )
    return scorer.score(
        reference,
        generated
    )["rougeL"].fmeasure

def meteor(reference, generated):
    return meteor_score(
        [reference.split()],
        generated.split()
    )

# ============================================================
# Evaluation
# ============================================================

def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    retriever = Retriever()

    results = []

    references = []
    generated_answers = []

    for item in dataset:

        question = item["question"]
        reference = item["reference_answer"]

        # Retrieve context
        retrieved = retriever.retrieve(question,top_k=K)
        context = retrieved["documents"]

        # Generate answer
        generated = generate_answer(
            question,
            context
        )

        references.append(reference)
        generated_answers.append(generated)

        results.append({
            "id": item["id"],
            "question": question,
            "reference_answer": reference,
            "generated_answer": generated,
            "BLEU": bleu(reference, generated),
            "ROUGE-L": rouge_l(reference, generated),
            "METEOR": meteor(reference, generated)
        })

    # BERTScore
    _, _, f1 = score(
        generated_answers,
        references,
        model_type="roberta-large",
        lang="en",
        verbose=False
    )

    for result, value in zip(results, f1.tolist()):
        result["BERTScore"] = value

    # Average scores
    n = len(results)

    averages = {
        "BLEU": sum(r["BLEU"] for r in results) / n,
        "ROUGE-L": sum(r["ROUGE-L"] for r in results) / n,
        "METEOR": sum(r["METEOR"] for r in results) / n,
        "BERTScore": sum(r["BERTScore"] for r in results) / n
    }

    # Print results
    print("\nGeneration Evaluation")
    print("---------------------")

    for metric, value in averages.items():
        print(f"{metric:<10}: {value:.4f}")

    # Save results
    output = {
        "metrics": averages,
        "results": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nResults saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
