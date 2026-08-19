# F_WAIT Completeness Filtering

## Goal

Identify F_WAIT samples whose player prefix is already a complete, naturally
answerable utterance. Filtering removes only manifest rows. Source manifests and
WAV files remain unchanged.

## Model Output

Fill these fields in `llm_requests.jsonl` without changing any other field:

```json
{
  "judgment": "complete|incomplete|uncertain",
  "confidence": "high|medium|low",
  "reason": "A short Chinese reason"
}
```

Definitions:

- `complete`: The shown player prefix is already a natural standalone statement,
  question, command, or exclamation that can be answered. It remains complete
  even if the hidden continuation could add more detail.
- `incomplete`: The prefix clearly stops inside a word, after a connector or
  function word, or at an unfinished grammatical/semantic constituent.
- `uncertain`: The available prefix and prior context do not support a stable
  decision. Do not use this merely to avoid choosing.

The request intentionally excludes the full query and hidden suffix.

## Hard Length Rule

Rows with `len(partial_query.strip()) >= 15` are not sent to the model. They are
recorded in `auto_drop_length_ge15.jsonl` with `decision=drop`.

After model filling, `complete` rows are dropped and `incomplete` rows are kept.
The handling of `uncertain` rows is intentionally deferred until manual review.
