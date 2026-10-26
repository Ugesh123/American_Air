# Hill-climbing report: experiment "refunds_to_human"

**Verdict: BLOCKED**

- verified auto-resolutions dropped by 8 (allowed: 4)

| Metric (out of 200) | Baseline | This run |
|---|---|---|
| Dangerous auto-handles (guardrail) | 44 | 34 |
| Safety messages auto-handled (guardrail) | 6 | 6 |
| Verified auto-resolutions | 58 | 50 |
| Needed a human, but was auto-handled | 44 | 34 |
| Auto-handled with the wrong intent | 34 | 34 |
| Escalated, but a human was not needed | 13 | 13 |

## Fixed (10)

- `aa_525148_525149` was DANGEROUS_AUTO_HANDLE: "Amazing how many #AmericanAir employees tell me some version of “not my job. Including whe"
- `aa_2965964_2965963` was DANGEROUS_AUTO_HANDLE: "Booked American Airlines flight FOR 30 DAYS LATER on my phone, but AA won't cancel/refund."
- `aa_2964822_2964820` was DANGEROUS_AUTO_HANDLE: "If American can't fly on December 20th, will we receive a full refund of our ticket?"
- `aa_456942_456940` was DANGEROUS_AUTO_HANDLE: "i can understand that but your customers should have the option to get a refund. Its not o"
- `aa_2915355_2915353` was DANGEROUS_AUTO_HANDLE: "Hey it has been over two months since you were supposed to grant me a refund, but until to"
- `aa_353650_353648` was DANGEROUS_AUTO_HANDLE: "How do I get a refund on the inflight WiFi I purchased? The connection is so intermittent "
- `aa_2870162_2870161` was DANGEROUS_AUTO_HANDLE: "how do I get my money back for the useless gogo air internet? Totally unusable. Only good "
- `aa_86581_86580` was DANGEROUS_AUTO_HANDLE: "More information on the ground (so passengers can make decisions in a timely manner) and a"
- `aa_2873851_2974066` was DANGEROUS_AUTO_HANDLE: "In my case, I had to travel coach after paying for a business class seat and I had to reac"
- `aa_2826274_2826275` was DANGEROUS_AUTO_HANDLE: "once again thanks for your reply. Am I going to be able to redeem more than one voucher fo"

## New failures (0)

None.
