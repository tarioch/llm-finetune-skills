# Comparing several base models

Use this when the question is which base model (or size) to fine-tune, not how to train one. Every candidate gets the
same data, settings and evaluation set, so the scores differ only by the model.

## Procedure

1. Settle once, for all candidates: the dataset, the evaluation set, the training settings (leave them at the
   defaults unless the user says otherwise) and the runs folder. Name the run folders after the model, e.g.
   `<runs>/<project>-3b` and `<runs>/<project>-7b`.
2. Run the skill's steps 2 to 5 for one candidate at a time, never in parallel: they share the GPU. Before each
   run, `ollama ps` must be empty (`ollama stop <model>` otherwise) and `preflight.py` must pass for that model's
   size. Do the smoke run (step 3) for every candidate: the chat template differs between model families, and the
   first example as rendered is where that shows.
3. Start with the smallest model. Its peak GPU memory and time per step tell whether the next size fits and how long
   it takes; say that to the user before starting the larger run.
4. Keep going if one candidate fails (out of memory, a template warning): note why in the summary and continue with
   the others.
5. Create an Ollama model per run with the `ollama-model` skill and compare them all in one `evaluate.py` call
   (several `--model`), with the same `--eval` and `--check`. Add the currently published model as the baseline,
   it is the one to beat.
6. Give the user one table: base model, train and validation loss, peak GPU memory, training time, GGUF size, score,
   median time per answer. Recommend one, with the trade-off (a model that scores one point more but is twice as
   large and slow is not necessarily the better one), and let the user decide what ships.
7. Clean up after the decision (step 6 of the skill, and `ollama rm` for the models that don't ship), asking first.

## What to compare

- Scores on a small evaluation set differ by noise: look at how many examples only one model got right, not only
  at the totals (`evaluate.py`'s report has that).
- Train and validation loss of different base models are not comparable (different tokenizers), only the evaluation
  score is.
