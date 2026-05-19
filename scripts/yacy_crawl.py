#!/usr/bin/env python3
"""Seed YaCy crawl jobs from config/search_sources.yml.

Examples:
  python scripts/yacy_crawl.py --dry-run --topic ai --limit 3
  python scripts/yacy_crawl.py --topic government
  YACY_USER=admin YACY_PASSWORD=yacy python scripts/yacy_crawl.py --id fda_press
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from requests.auth import HTTPDigestAuth


@dataclass(frozen=True)
class CrawlJob:
    source_id: str
    name: str
    crawl_url: str
    params: dict[str, Any]


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _selected_sources(
    config: dict[str, Any],
    *,
    ids: set[str] | None,
    topics: set[str] | None,
    cadence: str | None,
) -> list[dict[str, Any]]:
    defaults = config.get("defaults") or {}
    out: list[dict[str, Any]] = []
    for raw in config.get("sources") or []:
        if not isinstance(raw, dict):
            continue
        src = {**defaults, **raw}
        if not src.get("enabled", True):
            continue
        if src.get("backend") != "yacy":
            continue
        if ids and src.get("id") not in ids:
            continue
        source_topics = set(src.get("topics") or [])
        if topics and not source_topics.intersection(topics):
            continue
        if cadence and src.get("cadence") != cadence:
            continue
        out.append(src)
    return out


def _domain_regex(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.split("@")[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ".*"
    return rf"https?://([^/]+\.)?{re.escape(host)}(/|$).*"


def _recrawl_window(cadence: str) -> tuple[int, str]:
    if cadence == "hourly":
        return 1, "hour"
    if cadence == "weekly":
        return 7, "day"
    if cadence == "monthly":
        return 1, "month"
    return 1, "day"


def _bookmark_folder(cadence: str) -> str:
    if cadence in {"hourly", "daily", "weekly", "monthly"}:
        return f"/autoReCrawl/{cadence}"
    return "/crawlStart"


def _collection(src: dict[str, Any]) -> str:
    parts = [str(src["id"])]
    parts.extend(str(topic) for topic in src.get("topics") or [])
    return ",".join(dict.fromkeys(parts))


def _crawl_job(src: dict[str, Any]) -> CrawlJob:
    crawl_url = src.get("crawl_url") or src.get("rss_url") or src.get("base_url")
    if not crawl_url:
        raise ValueError(f"source {src.get('id', '<missing id>')} has no crawl_url/base_url")

    cadence = str(src.get("cadence", "daily"))
    recrawl_n, recrawl_unit = _recrawl_window(cadence)
    must_match = src.get("must_match") or _domain_regex(src.get("base_url") or crawl_url)
    store_cache = bool(src.get("store_cache", False))

    params: dict[str, Any] = {
        "crawlingstart": "Start New Crawl",
        "crawlingMode": "url",
        "crawlingURL": crawl_url,
        "crawlingDepth": int(src.get("crawl_depth", 1)),
        "crawlingDomMaxPages": int(src.get("max_pages_per_domain", 200)),
        "range": src.get("scope", "domain"),
        "mustmatch": must_match,
        "mustnotmatch": src.get("must_not_match", ""),
        "indexmustmatch": src.get("index_must_match", ".*"),
        "indexmustnotmatch": src.get("index_must_not_match", ""),
        "crawlingQ": "on" if src.get("allow_query_urls", False) else "off",
        "indexText": "on",
        "indexMedia": "off",
        "crawlOrder": "off",
        "xsstopw": "on",
        "storeHTCache": "on" if store_cache else "off",
        "cachePolicy": src.get("cache_policy", "iffresh"),
        "recrawl": "reload",
        "reloadIfOlderNumber": recrawl_n,
        "reloadIfOlderUnit": recrawl_unit,
        "crawlingIfOlderCheck": "on",
        "crawlingIfOlderNumber": recrawl_n,
        "crawlingIfOlderUnit": recrawl_unit,
        "createBookmark": "on",
        "bookmarkFolder": _bookmark_folder(cadence),
        "bookmarkTitle": src.get("name", src["id"]),
        "collection": _collection(src),
    }
    return CrawlJob(
        source_id=str(src["id"]),
        name=str(src.get("name", src["id"])),
        crawl_url=str(crawl_url),
        params=params,
    )


def _start_job(
    session: requests.Session,
    *,
    base_url: str,
    job: CrawlJob,
    user: str | None,
    password: str | None,
    timeout_s: float,
) -> requests.Response:
    url = f"{base_url.rstrip('/')}/Crawler_p.html"
    auth: Any = HTTPDigestAuth(user, password) if user and password else None
    resp = session.get(url, params=job.params, auth=auth, timeout=timeout_s)
    if resp.status_code == 401 and user and password:
        resp = session.get(url, params=job.params, auth=(user, password), timeout=timeout_s)
    resp.raise_for_status()
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="config/search_sources.yml")
    parser.add_argument("--base-url", default=os.environ.get("BANNA_YACY_BASE_URL", "http://localhost:8090"))
    parser.add_argument("--id", dest="ids", action="append", help="Only crawl this source id.")
    parser.add_argument("--topic", action="append", help="Only crawl sources with this topic.")
    parser.add_argument("--cadence", choices=["hourly", "daily", "weekly", "monthly"])
    parser.add_argument("--limit", type=int, help="Limit number of jobs submitted.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without calling YaCy.")
    parser.add_argument("--user", default=os.environ.get("YACY_USER", "admin"))
    parser.add_argument("--password", default=os.environ.get("YACY_PASSWORD", "yacy"))
    parser.add_argument("--no-auth", action="store_true")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    config = _load_config(args.sources)
    sources = _selected_sources(
        config,
        ids=set(args.ids) if args.ids else None,
        topics=set(args.topic) if args.topic else None,
        cadence=args.cadence,
    )
    if args.limit is not None:
        sources = sources[: args.limit]
    jobs = [_crawl_job(src) for src in sources]

    if not jobs:
        print("no matching YaCy sources")
        return 0

    session = requests.Session()
    user = None if args.no_auth else args.user
    password = None if args.no_auth else args.password

    for job in jobs:
        if args.dry_run:
            print(f"DRY {job.source_id}: {job.crawl_url}")
            print(f"  depth={job.params['crawlingDepth']} cadence={job.params['bookmarkFolder']}")
            print(f"  mustmatch={job.params['mustmatch']}")
            continue
        resp = _start_job(
            session,
            base_url=args.base_url,
            job=job,
            user=user,
            password=password,
            timeout_s=args.timeout,
        )
        print(f"STARTED {job.source_id}: HTTP {resp.status_code} {job.crawl_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
