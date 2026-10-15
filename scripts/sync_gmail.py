#!/usr/bin/env python
"""Explicit one-shot Gmail read-only sync into Jarvis's local evidence index."""
from __future__ import annotations

import argparse
import sys

from app.core import config, storage
from app.core.gmail import configured_connector, sync_gmail_full, sync_gmail_incremental, GmailAPIError
from app.core.gmail_oauth import GmailConfigurationError, GmailOAuthError
from app.core.external_sources import ExternalSourceValidationError
from app.core.gmail_sync_state import GmailFullSyncRequired


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Manually sync Gmail messages as local searchable evidence.")
    parser.add_argument("--max-pages", type=int, default=None, help="optional safety limit for pages fetched")
    parser.add_argument("--incremental", action="store_true", help="apply Gmail History changes from the last successful full/incremental sync")
    args = parser.parse_args(argv)
    if args.max_pages is not None and args.max_pages < 1:
        parser.error("--max-pages must be positive")
    if args.incremental and args.max_pages is not None:
        parser.error("--max-pages applies only to full sync")
    db = None
    try:
        connector = configured_connector()
        # Complete OAuth before opening the configured database. This also
        # fails early when local OAuth configuration is missing.
        connector.client.oauth.access_token()
        db = storage.JarvisDB(config.db_path())
        from app.core.external_sources import ExternalSourceRepository
        repository = ExternalSourceRepository(db)
        from app.core.gmail_sync_state import GmailSyncStateRepository
        state = GmailSyncStateRepository(db)
        if args.incremental:
            summary = sync_gmail_incremental(connector, repository, state)
            mode = "incremental"
        else:
            summary = sync_gmail_full(connector, repository, state, max_pages=args.max_pages)
            mode = "full"
        print(f"Gmail {mode} sync complete: " + ", ".join(f"{key}={value}" for key, value in summary.items()))
        return 0
    except (GmailConfigurationError, GmailOAuthError, ExternalSourceValidationError,
            GmailFullSyncRequired, GmailAPIError, ValueError, RuntimeError) as exc:
        print(f"Gmail sync failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Provider/library internals can contain private response data. Never
        # echo unexpected exception details from the sync operation.
        print("Gmail sync failed due to an unexpected local or provider error.", file=sys.stderr)
        return 1
    finally:
        if db is not None:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
