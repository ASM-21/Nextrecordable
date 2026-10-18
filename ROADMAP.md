# Roadmap

## Now

- **First real-data run.** `nextrecordable run-all` on the 2016 onward ITA files, with and without DOL enforcement files. Publish `reports/METRICS.md`, including where the models fail to beat their baselines.
- Re-tune matcher weights against real DOL to ITA links if the `inspection_link_share` gate warns.

## Next

- Accounts and metered tiers for the batch API (free lookups, paid books of business).
- One-page PDF report per establishment for brokers to hand to clients.
- Container image and a hosted instance behind TLS.
- DOL `osha_accident` features; ITA case-detail (300/301) injury mix.

## Later

- BLS SOII industry benchmarks next to the NAICS-4 pooled rates.
- Narrative text features from 301 case detail once enough years exist.
- Read-only multi-process serving.
