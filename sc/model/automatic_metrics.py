"""How automatic metrics were calculated for the Social Chemistry 101 paper.

This isn't hooked up directly to the generation scripts right now, but it's provided
as a reference so you know which versions of bleu, rouge, etc., we used.
"""

import os
import json
import numpy as np

from nltk import bleu
from rouge import Rouge
from collections import defaultdict
from nltk.translate.bleu_score import SmoothingFunction

smoothing = SmoothingFunction().method1
weights = [0.25] * 4
rouge = Rouge()


print(
    "\t".join(["Task", "LM", "Macro perplexity", "Micro perplexity", "BLEU", "ROUGE"])
)

for setup in ["action_all", "rot_all", "action_from_rot"]:
    for lm in [
        "openai-gpt",
        "gpt2",
        "gpt2-xl",
        "bart-large",
        "t5-large",
        "random_gpt2-xl",
    ]:
        # Get the perplexity results
        test_results = f"output/{setup}_{lm}/test_results.txt"
        macro_perplexity, micro_perplexity = 0.0, 0.0

        if os.path.exists(test_results):
            with open(test_results) as f_in:
                for line in f_in:
                    if line.startswith("macro_perplexity = "):
                        macro_perplexity = float(line[len("macro_perplexity = ") :])
                    elif line.startswith("micro_perplexity = "):
                        micro_perplexity = float(line[len("micro_perplexity = ") :])

        # Compute BLEU and ROUGE from the text predictions
        test_predictions = f"output/{setup}_{lm}/test_predictions_p0.9.jsonl"
        bleu_score, rouge_score = 0, 0

        if os.path.exists(test_predictions):
            data = [json.loads(line.strip()) for line in open(test_predictions)]
            gold = defaultdict(list)
            predictions = defaultdict(list)

            for ex in data:
                curr_gold = ex["output"].lower().replace("<eos>", "").strip()
                curr_pred = ex["prediction"].lower()

                if len(curr_gold) > 0 and len(curr_pred) > 0:
                    gold[ex["input"]].append(curr_gold)
                    predictions[ex["input"]].append(curr_pred)

            bleu_scores, rouge_scores = [], []

            for input, curr_gold in gold.items():
                curr_predictions = list(predictions[input])

                # The refs and gold must be in the same size
                length = min(len(curr_gold), len(curr_predictions))

                if length > 0:
                    hyps = curr_predictions[:length]
                    refs = curr_gold[:length]

                    rouge_scores.extend(
                        [
                            score["rouge-l"]["f"]
                            for score in rouge.get_scores(hyps, refs)
                        ]
                    )

                    hyps = [tuple(h.split()) for h in hyps]
                    refs = [tuple(r.split()) for r in refs]
                    bleu_scores.extend(
                        [
                            bleu(
                                refs,
                                pred,
                                weights=weights,
                                smoothing_function=smoothing,
                            )
                            for pred in hyps
                        ]
                    )

                    bleu_score = 100.0 * np.mean(bleu_scores)
                    rouge_score = 100.0 * np.mean(rouge_scores)

        print(
            "\t".join(
                [
                    setup,
                    lm,
                    f"{macro_perplexity:.3f}" if macro_perplexity > 0 else "-",
                    f"{micro_perplexity:.3f}" if micro_perplexity > 0 else "-",
                    f"{bleu_score:.3f}" if bleu_score > 0 else "-",
                    f"{rouge_score:.3f}" if rouge_score > 0 else "-",
                ]
            )
        )
