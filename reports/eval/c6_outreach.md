# C6 · Outreach writer: evaluation

20 Djinni JDs (seed 42) → C1 → C3 → C4 → C5 → top-3 messages per job.

## A) Message quality

| writer | messages | faithful (0 unsupported) | claims checked / msg | verified personal facts / msg | length in range | mean chars | CTA present | same-job trigram Jaccard (lower=better) | fell back to template |
|---|---|---|---|---|---|---|---|---|---|
| template (baseline) | 59 | 1.000 | 7.830 | 5.339 | 1.000 | 530.746 | 1.000 | 0.459 | n/a |

## B) Claim verifier: hallucination injection

One fabricated sentence of each type is inserted into each template message; the verifier should flag it.

| injected claim | n | detected (recall) |
|---|---|---|
| fake skill | 59 | 1.000 |
| wrong years | 59 | 0.881 |
| fake quote | 59 | 1.000 |
| fake domain | 59 | 1.000 |
| none (clean message) → false-positive rate | 59 | 0.000 |

## C) LLM-as-judge (rubric: eval/rubrics/outreach.md)

_Not run: no OPENROUTER_API_KEY._

## Example messages

```
Hi there,

I came across your profile as UI/UX DESIGNER with 4 years of experience.
Your background in Figma, UI/UX, CSS is exactly what caught my attention.
In particular, this line stood out: "Familiarity: I design conventional layouts and ensure they seem less complex to users just because they’re familiar".

We are hiring for a Trainee Visual / Interaction Designer.
The role needs strong Figma, UI/UX.

Would you be open to a short 15-minute call this week to talk it through?

Best regards
```

```
Hi there,

I came across your profile as UI/UX Designer with 1.5 years of experience.
Your background in Figma, UI/UX, Agile is exactly what caught my attention.
In particular, this line stood out: "Interactive wireframing in Axure and Figma;".

We are hiring for a Trainee Visual / Interaction Designer.
The role needs strong Figma, UI/UX.
You mentioned you are looking for "I am looking for a design team who creates modern solutions, close collaboration with the development team, active participation in…", so I think this could be a good match.

Would you be open to a short 15-minute call this week to talk it through?

Best regards
```
