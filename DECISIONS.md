# Decisions log

Judgment calls made while taking nextrecordable to real data (November 2026), newest last. Each says what was decided, why, and what would change it.

## Starting point

1. **The repo differs from the handoff notes, and the repo wins.** Notes put the last commit at f1d99af with about 137 tests, plus billing (API keys, tier metering, a Stripe webhook), a pricing page with Stripe Payment Links, `/score-pdf` and Docker. The repo at `ebc687b` has 100 tests and none of those; splink had already been replaced by a fixed-weight matcher. No other branch holds them. So "remaining launch work" here includes building compact versions of keys, metering, the webhook, a pricing page and Docker. Prices follow the notes (Free, Pro $79/mo, Team $499/mo, Enterprise).

## Data

2. **ITA files from a third-party copy, clearly labeled.** osha.gov's CDN refused every request from the build machine. A search for the exact file names found a public GitHub repo (a 2024 student project) holding OSHA's 300A CSVs for 2019 to 2023 with the official column layout. I used them rather than stop, because a real run with a labeled caveat beats no real run. Safeguards: `data/raw/SOURCES.json` records origin, commit and snapshot; `provenance.py` puts a notice in every API response, PDF and METRICS.md until official files replace them; DATA-GAPS.md gives the swap in under an hour.
3. **DOL enforcement from data.dol.gov bulk files.** The old catalog (enforcedata.dol.gov) now redirects to data.dol.gov, whose React app hides a predictable bulk URL (`/data-catalog/OSHA/<table>/OSHA_<table>.zip`). Loaded from 2006 onward; the panel looks back at most ten years.
4. **The directive text was read from Hawaii's republished copy** (HIOSH adopted CPL 02-01-067 in its entirety and includes the federal cover page). Quotes in `data/sst_cpl_02_01_067.json` are verbatim from it.

## Data quality rules (each has a reproduction test)

5. **Implausible exposure gets no rate.** Employees between 1 and 100,000 and hours per employee between 100 and 8,760 (hours in a year). About 1.7% of establishment-years fail. They keep their counts; their hours and rates are NULL. Rejecting rather than capping, because a capped typo is still a made-up number.
6. **Implausible case counts get no rate and are never a training label.** More than five cases and more than two per employee. 223 rows; the threshold is loose on purpose so a genuinely bad year at a small site is kept.
7. **Identifier cleanup.** Trailing ".0"/".00" removed from IDs, NAICS, size and establishment type; four timestamp formats parsed.

## Models

8. **Serve the credibility estimate, not the boosted rate model,** whenever the boosted model fails to beat it on the holdout year. On real data it lost (-7.1% deviance skill in 2022, a tie in 2021). Both are still reported. The rule is automatic, so a better model on official data will be served without a code change.
9. **Kept the binary model.** It beats size + NAICS within every size band, on unseen establishments and on a second test year (FINDINGS.md).

## SST exposure report

10. **No numeric cutoff for the high-rate list.** The directive says separate manufacturing and non-manufacturing DART cutoffs exist and publishes neither. The report gives percentile position within each group and says plainly that OSHA does not publish the cutoff.
11. **Upward trending: our reading is labeled as ours.** "Twice the private sector national average in CY 2022" does not say which rate. Read as DART, with the BLS 2022 private-industry DART rate 1.7, so 3.4, and "trend upward annually" as strictly rising 2021 < 2022 < 2023. Both conditions are shown separately.
12. **"No filing found", never "non-responder".** Year-over-year churn is 30 to 33% in complete years, so a missing filing is usually a rename, closure or move. The report gives the measured churn next to the flag.
13. **Scope.** In comparison means: filed for CY 2023, 20 or more employees, NAICS not 23, a usable rate, and not marked state or local government (ITA establishment_type 2 or 3). Federal agencies are not flagged by ITA's type codes and stay in.
14. **36-month deletion context** uses DOL `insp_scope = 'A'` (complete inspection) as "comprehensive". Inspections are linked by name and address, so the report says a missing record is not proof of none.
15. **No Form 301 data anywhere in the report path**, enforced by a test on the module's code.
16. **Example report on a real establishment kept out of git.** `reports/examples/` is gitignored. The establishment's data is OSHA-published, but the README's own rule is no republishing scores about named establishments without legal review, and the repo is public.

## Launch plumbing

17. **Accounts in SQLite, separate from DuckDB.** The analytics file is rebuilt on every data refresh; accounts must survive it. Keys are stored as SHA-256 hashes, shown once in the email.
18. **Quotas are first guesses:** Free 100 lookups a month and no batch, Pro 2,000 and 1,000-row batches, Team 25,000 and 5,000-row batches. Anonymous lookups: 20 a day per IP, counted in memory (resets on restart, which is acceptable for one process). All in `data/tiers.json`.
19. **Email is dry-run by default** (writes .eml files to `data/outbox`). SMTP is the only live provider, because every transactional service offers it; an HTTP provider can be added behind the same `Mailer` interface.
20. **Stripe Payment Links map to tiers by link id** through `NEXTRECORDABLE_STRIPE_LINK_TIERS`. Webhook events are processed once each; cancellation drops to Free rather than revoking the key.
21. **Caddy for TLS** on one host: automatic certificates, about ten lines of config. Not tested here (no Docker daemon in the build environment); the server's startup path was smoke-tested directly.

## Process

22. One DuckDB segmentation fault during a build that reused inspection links. Every step succeeded run alone, and the full build succeeded on retry. Not reproduced; noted in case it recurs.
23. Commits carry Andrew's noreply identity. No secrets, keys, customer emails or local paths are in the repo; `deploy/env.example` holds placeholders only.

## After the first merge

24. **The credibility constant stays at 200,000 hours.** Fitting it on the validation year picks about 100,000 hours, which does worse on the holdout year; holdout deviance is flat between roughly 150,000 and 300,000. A hand-set value that sits on a flat optimum beats a fitted one that chases one year.
25. **Drivers only for a served boosted model.** With the experience-rated estimate served, `/score` shows no drivers and says why, instead of showing binary-model log-odds under a rate heading.
26. **Batch SST and region summaries for extension centers.** `/sst-batch` reuses the single-report logic (a test checks they agree). `region-summary` publishes counts only and suppresses counts from 1 to 4, so a regional table cannot point at one plant.
27. **Link precision was labeled by Claude**, not a second person: 80 of 80 sampled fuzzy links were the same establishment. A human spot-check of the closest calls in `reports/link_audit.json` would make it stronger.
28. **Key recovery by emailed link**, never by showing a key in the browser. The request answers the same way for unknown addresses, is capped at five a day per client, and the link is single use and expires in an hour. Usage stays with the account.
29. **Container check in CI.** No Docker daemon was available while writing the Dockerfile, so CI builds the image and starts it on a synthetic database on every push.
