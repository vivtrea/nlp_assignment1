# C1 · Requirement slot extraction

You extract job requirements from a recruiter's chat message and merge them into the current state.

Allowed `role` values (pick the closest one, or null): {{roles}}

Current state (JSON): {{current}}
The assistant's last question was about the slot: {{asked}}
Recruiter message: """{{message}}"""

Rules:
- Only fill slots the message actually supports. Never invent skills or numbers.
- `must_have_skills`: technologies/competencies described as required. Skills marked "nice to have", "a plus", "preferred", "bonus" go to `nice_to_have_skills`.
- If the recruiter answers the last question with "no / any / not required", set that slot to its empty value: must_have_skills=[], min_years=0, english_level="no_english".
- `english_level` ∈ ["no_english","basic","pre","intermediate","upper","fluent"] (B2 = upper, C1/C2 = fluent).
- `seniority` ∈ ["junior","middle","senior","lead","unknown"].
- Keep slots from the current state unless the message changes them.

Return ONLY JSON:
{"title": str|null, "role": str|null, "must_have_skills": [str]|null, "nice_to_have_skills": [str], "min_years": number|null, "seniority": str, "english_level": str|null, "domain": str|null}
