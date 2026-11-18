# Roadmap

## Now

- **Official ITA files.** Replace the third-party copies with osha.gov downloads for 2016 to 2024 (`DATA-GAPS.md`), retrain, and update `FINDINGS.md`.
- **Launch.** `deploy/RUNBOOK.md`, in order: legal read, domain, host, DNS, Stripe test mode, email provider, Stripe live.
- Measure precision of fuzzy DOL-to-ITA links on a hand-labeled sample, then revisit the `inspection_link_share` gate threshold.

## Next

- A successor SST directive (CPL 02-01-067 runs to about May 2027): add its data file when issued.
- A rate model that beats experience rating on the holdout year, or drop the boosted model entirely.
- DOL `osha_accident` features (already downloaded, not used).
- Self-serve key reissue and a customer portal link from `/account`.

## Later

- BLS SOII industry benchmarks next to the NAICS-4 pooled rates.
- ITA case-detail (300/301) injury mix as model features only, never in reports.
- Read-only multi-process serving.
