# Local Search With YaCy

This project treats YaCy as a curated local crawl/cache for trusted sources,
not as a full Google/Bing replacement. The agent-facing tool remains `search`;
YaCy is selected with `backend: "yacy"` or `BANNA_SEARCH_BACKEND=yacy`.

## Run YaCy

```bash
docker compose -f compose.yacy.yml up -d
```

The UI is available at:

```text
http://localhost:8090
```

The Compose file binds YaCy to `127.0.0.1` only. The Docker image defaults to
`admin:yacy`; change that before exposing YaCy beyond this machine.

YaCy itself defaults to peer-capable behavior. The agent tool forces
`resource=local`, `verify=cacheonly`, and a trusted-domain `urlmaskfilter`
built from `config/search_sources.yml`. Avoid using raw YaCy global search for
agent workflows unless you explicitly want peer-network results in the local
index.

## Seed Crawls

Preview a few sources:

```bash
python scripts/yacy_crawl.py --dry-run --topic ai --limit 5
```

Start one source:

```bash
python scripts/yacy_crawl.py --id fda_press
```

Start a topic bucket:

```bash
python scripts/yacy_crawl.py --topic government
```

Finance/quant crawl previews:

```bash
python scripts/yacy_crawl.py --dry-run --topic finance_quant --limit 10
python scripts/yacy_crawl.py --dry-run --id sec_filings
```

Finance/quant crawl starts:

```bash
python scripts/yacy_crawl.py --topic finance_quant --cadence daily
python scripts/yacy_crawl.py --topic finance_quant --cadence hourly
python scripts/yacy_crawl.py --topic finance_quant --cadence weekly
```

The source registry is `config/search_sources.yml`. Each YaCy source has:

- `source_tier`
- `topics`
- `crawl_url`
- `cadence`
- `crawl_depth`
- `max_pages_per_domain`
- `api_preferred` when an official structured API should win over crawled pages

## Agent Use

For local curated cache search:

```python
from banna_agent.tools.search import search

out = search(
    "FDA obesity drug approval",
    backend="yacy",
    source_filter=r".*fda\.gov.*",
    since="2026-01-01",
)
```

By default, `backend="yacy"` applies a trusted-domain URL mask from
`config/search_sources.yml`. Override it with `source_filter` for narrower
queries, or with `MYAGENT_YACY_URLMASK` for a process-wide mask.

For the CLI:

```bash
BANNA_SEARCH_BACKEND=yacy myagent
```

Use live backends for broad discovery or breaking news when YaCy is thin or
stale.

## Storage Estimates

Actual size depends heavily on crawl depth, whether YaCy stores the HTTP cache,
PDF volume, media indexing, and duplicate pages. With `store_cache: false` and
text indexing only, use these planning numbers:

| Profile | Approx docs | Disk target | RAM target | Use case |
|---|---:|---:|---:|---|
| Tiny pilot | 5k-25k | 1-5 GB | 1 GB | Validate source quality |
| Small curated | 25k-100k | 5-20 GB | 1-2 GB | Daily official/science/AI/news cache |
| Medium curated | 100k-500k | 20-80 GB | 2-4 GB | More domains, hourly news buckets |
| Large local cache | 500k-2M+ | 80-300+ GB | 4-8+ GB | Serious local search appliance |

Start with a 20-40 GB budget. Expand only after checking result quality,
staleness, and crawl noise.

## Cadence Tradeoffs

Hourly:

- Pros: better for geopolitics, markets, and fast-moving news.
- Cons: higher bandwidth, more duplicate churn, more risk of hitting crawl
  limits, and more stale partial pages.

Daily:

- Pros: best default for official agencies, science, AI/vendor blogs, biotech,
  and finance sources.
- Cons: misses intraday developments.

Weekly:

- Pros: low cost for slower analysis sites and reference-like sources.
- Cons: too stale for policy, markets, and active conflicts.

On-demand:

- Pros: lowest disk and bandwidth cost.
- Cons: bad agent latency and no useful local cache when the user asks.

Initial recommendation: hourly only for AP/Reuters/BBC/NPR/Al Jazeera/CNBC
style news buckets; daily for official, science, AI, biotech, and VC sources;
weekly for analysis sources after the first crawl settles.

## Routing Rules

Prefer official APIs for:

- SEC filings and company disclosures
- PubMed/NCBI literature
- ClinicalTrials.gov trials
- openFDA data
- Federal Register, govinfo, Congress.gov, USAspending records
- Wikipedia/MediaWiki reference facts

Use YaCy for:

- Crawled trusted news/source pages
- Vendor and agency newsrooms
- RSS/topic pages
- Local cache fallback
- Private/domain-specific sources

If YaCy returns thin or old results, the agent should say so and call a better
backend.

## Finance And Quant

The source registry includes a dedicated `finance_quant` topic for market
notices, macro releases, official datasets, and quant/institutional research.
YaCy should crawl these for discovery and citation context, but numeric
modeling should use structured APIs whenever available.

API-preferred structured targets:

- SEC EDGAR for filings, submissions, and XBRL/company facts
- FRED/ALFRED for macroeconomic time series and vintages
- U.S. Treasury XML feeds for yield/rate data
- BLS API for CPI, employment, wages, and labor series
- BEA API for GDP, PCE, and national accounts
- EIA API for energy and commodity series
- CFTC COT data for futures positioning

YaCy crawler targets:

- SEC structured-disclosure RSS feed page (`sec_filings`)
- Federal Reserve feeds and FEDS working papers
- Treasury, BLS, BEA, EIA, and CFTC release pages
- Nasdaq RSS, NYSE notices, Cboe equities/options notices, CME notices
- arXiv q-fin recent papers
- AQR, Man Institute/Numeric, Research Affiliates, Robeco, Quantpedia,
  Alpha Architect, Dimensional, MSCI, S&P Indexology, and FTSE Russell

Recommended cadence:

- Hourly: SEC filing feed page, market/news buckets, exchange notices
- Daily: Fed/Treasury/BLS/BEA/EIA release pages, arXiv q-fin, institutional
  research that updates regularly
- Weekly: CFTC COT pages and slower long-form quant research

## References

- YaCy Docker install: https://www.yacy.net/download_installation/
- YaCy search API: https://wiki.yacy.net/index.php/Dev:APIyacysearch
- YaCy crawler API: https://yacy.net/api/crawler/
