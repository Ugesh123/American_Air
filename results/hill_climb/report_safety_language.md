# Hill-climbing report: experiment "safety_language"

**Verdict: IMPROVED**

- All guardrails held.

| Metric (out of 200) | Baseline | This run |
|---|---|---|
| Dangerous auto-handles (guardrail) | 44 | 38 |
| Safety messages auto-handled (guardrail) | 6 | 0 |
| Verified auto-resolutions | 58 | 58 |
| Needed a human, but was auto-handled | 44 | 38 |
| Auto-handled with the wrong intent | 34 | 34 |
| Escalated, but a human was not needed | 13 | 13 |

## Fixed (6)

- `aa_2344855_2344853` was DANGEROUS_AUTO_HANDLE: "Coming back was better. The staff in Lexington was great. Didn’t have issues until I lande"
- `aa_1426561_1426559` was DANGEROUS_AUTO_HANDLE: "[USER] [USER] which one of you is going to draw the line on the flying kennels you have al"
- `aa_2664120_2664118` was DANGEROUS_AUTO_HANDLE: "[USER] And then, plane staff harassed him further when he was told there was no bin space "
- `aa_471017_471018` was DANGEROUS_AUTO_HANDLE: "10 years ago, as an unaccompanied minor flying with aa, I was sexually assaulted by a pass"
- `aa_2907420_2907418` was DANGEROUS_AUTO_HANDLE: "One of the passenger requested her name and once she knew she started to threat the passen"
- `aa_2282589_2282595` was DANGEROUS_AUTO_HANDLE: "Then this could've all been handled but she went and lied to the other flight attendants s"

## New failures (0)

None.
