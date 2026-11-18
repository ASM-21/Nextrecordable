# Runbook

One small host, Docker Compose, Caddy for TLS. Everything here is prepared and nothing is live: no domain is bought, no DNS record exists, Stripe is in test mode, and email is dry-run.

## Launch checklist (Andrew, in order)

Steps marked **$** cost money or change something public, and none have been done.

1. **Finish the data.** Follow `DATA-GAPS.md` on a machine that can reach osha.gov, so no output carries the third-party-copy notice. Commit the new `reports/`.
2. **Legal read.** Have someone read the README's "Read before relying on a score", the SST report PDF (`nextrecordable` writes one for any establishment through `/sst-exposure-pdf`) and the pricing page fine print. The SST report names real employers; decide whether reports about an establishment are sold only to that establishment, or to anyone.
3. **$ Buy the domain.** Any registrar.
4. **$ Rent the host.** 1 to 2 vCPU, 2 GB RAM, 25 GB disk is enough for the API: the database is 2.3 GB on disk, and the server measured about 0.4 GB of memory after startup with lookups answering in about 0.14 s. Build and train on a bigger machine (the full build peaked near 11 GB) and copy the results over. Ubuntu 24.04 LTS with Docker Engine and the compose plugin. Open ports 22, 80 and 443 only.
5. **Copy code and data.** `git clone` the repo on the host, then copy `data/nextrecordable.duckdb` and `data/model/` from the machine that built them into `data/` beside the code (rsync over SSH; the database never goes through git).
6. **$ DNS.** Add an A record (and AAAA if the host has IPv6) for the chosen name pointing at the host. Wait for it to resolve before step 8, or Caddy's certificate request fails and retries.
7. **Configure.** `cp deploy/env.example deploy/.env` on the host and fill it in. Leave `NEXTRECORDABLE_MAIL_PROVIDER=dryrun` and the Stripe test values for now.
8. **Start.**

```bash
cd deploy && docker compose --env-file .env up -d --build
```

Then check `https://<domain>/health` (should show `"status": "ok"`), `/pricing` and the lookup page.

9. **Stripe in test mode.** Create Payment Links for Pro and Team (recurring monthly). Put each link URL in `NEXTRECORDABLE_PAYMENT_LINK_PRO` / `_TEAM`, and map each link id (`plink_...`) to its tier in `NEXTRECORDABLE_STRIPE_LINK_TIERS`. Add a webhook endpoint `https://<domain>/stripe/webhook` for `checkout.session.completed` and `customer.subscription.deleted`, and put its signing secret in `NEXTRECORDABLE_STRIPE_WEBHOOK_SECRET`. Restart with `docker compose up -d`. Buy Pro with a Stripe test card and confirm a key email appears in `data/outbox/`, then that the key works on `/sst-exposure-pdf`.
10. **$ Email provider.** Pick one with SMTP (Postmark, SES, Resend, Mailgun), verify the sending domain (more DNS records: SPF, DKIM), fill in the `NEXTRECORDABLE_SMTP_*` values, set `NEXTRECORDABLE_MAIL_PROVIDER=smtp`, restart, and repeat the test purchase to a real inbox.
11. **$ Stripe live mode.** Recreate the Payment Links and the webhook in live mode, replace the four Stripe values in `.env`, restart. Make one real purchase and refund it.
12. **Tell people.** The MRC pitch and the broker channel.

## Routine operations

**Logs.** `docker compose logs -f app` (requests) and `docker compose logs caddy` (certificates).

**Restart after an `.env` change.** `docker compose up -d` (it recreates only what changed).

**Deploy new code.**

```bash
git pull && cd deploy && docker compose up -d --build
```

**Back up accounts.** `data/accounts.sqlite` is the only file that cannot be rebuilt. Copy it off the host daily (for example a cron job running `sqlite3 data/accounts.sqlite ".backup data/accounts-$(date +%F).sqlite"` followed by an off-host copy).

**Refresh data (yearly).** ITA data for a calendar year is due March 2 of the next year, and OSHA publishes it some months later. When a new year appears: add its URL to `config.py`, rebuild and retrain on the build machine (DATA-GAPS.md steps 3 to 7), copy the new `nextrecordable.duckdb` and `model/` to the host, and restart the app. Accounts are untouched.

**Watch for a new SST directive.** CPL 02-01-067 runs two years from May 20, 2025 unless replaced. When OSHA issues a successor, add a new data file beside `nextrecordable/data/sst_cpl_02_01_067.json` with its quotes, data years and source, and point `sst.DIRECTIVE_FILE` at it. Until then every report prints the directive's dates, so a stale report is visibly stale.

**Reissue a lost or leaked key.** On the host:

```bash
docker compose exec app python -c "from nextrecordable.accounts import AccountStore; s=AccountStore('/app/data/accounts.sqlite'); print(s._db.execute('select id,email,tier,key_prefix,active from accounts').fetchall())"
```

Deactivate the old account (`s.deactivate(id)`), create a new one with the same email, tier and Stripe customer (`s.create(email, tier, stripe_customer=...)`), and send the new key with `nextrecordable.mailer.key_email`.

## When something is wrong

| symptom | look at | likely fix |
|---|---|---|
| site down, Caddy log shows ACME errors | DNS for the domain | the A record is missing or points elsewhere |
| `/health` 502 from Caddy | `docker compose logs app` | app still loading (about a minute), or `data/` missing or unreadable by uid 10001 |
| paid customer got no email | `data/outbox/` (dry-run) or provider dashboard; webhook delivery log in Stripe | provider still `dryrun`, or `NEXTRECORDABLE_STRIPE_LINK_TIERS` lacks the link id (the webhook answers `ignored`) |
| webhook 400 "Bad signature" | Stripe webhook secret | test and live secrets differ; replace and restart |
| 429 for a customer | `GET /account` with their key | monthly quota reached; upgrade tier or raise the quota in `data/tiers.json` and redeploy |
