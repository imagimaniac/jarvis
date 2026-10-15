# External source integrations

Jarvis stores imported external records locally as source evidence. It does not
create canonical memories from imported content. Records are indexed through
`external_source_records` and immutable snapshots in `external_source_versions`;
the existing `entries` table receives a searchable projection for SQLite FTS,
recall, citations, and `/api/search`. Current versions are searchable and older
versions remain available for provenance. External records never enter personal
context automatically.

## Provider-neutral foundation

`app/core/external_sources.py` defines the normalized record, connector page,
repository, and `ingest_connector_page()` boundary. A connector returns a
`SyncPage` of credential-free `NormalizedExternalRecord` values; the repository
upserts by provider and external ID, maintains content hashes/versions, and
projects current content into existing evidence search. Cursors are passed in
memory and are not persisted by this foundation.

The legacy user-triggered Takeout importer is separate and unchanged.

## Gmail read-only connector

Gmail's provider implementation is isolated in `app/core/gmail.py`, with the
local installed-app OAuth flow in `app/core/gmail_oauth.py`. It uses Gmail API
`users.messages.list` and `users.messages.get` (`format=full`), follows page
tokens, retries transient errors, and skips messages deleted between listing
and retrieval. Message bodies are normalized from plain text or HTML MIME parts;
attachments are not imported. The manual full sync remains the initial setup and
recovery operation. Successful full syncs save a local Gmail History API
checkpoint; subsequent explicit incremental syncs fetch History pages and only
re-read messages changed since that checkpoint. There is no daemon, timer, push
subscription, or automatic email monitoring.

### Google Cloud setup

1. Create/select a Google Cloud project and enable the **Gmail API**.
2. Configure the OAuth consent screen for the intended account. Keep it in
   testing while developing and add the account as a test user if Google
   requires it.
3. Create an OAuth client with application type **Desktop app** and download
   its client-secrets JSON file to a private local location. Do not place the
   file in source control. Jarvis uses the installed-app loopback callback on
   `127.0.0.1` and opens the authorization page in the local browser.
4. Configure the exact read-only scope:

   `https://www.googleapis.com/auth/gmail.readonly`

   Jarvis does not request Gmail modify, send, or delete scopes.

### Local configuration and authorization

Set either the client-secrets file path:

```sh
export JARVIS_GMAIL_CLIENT_SECRETS_FILE="$HOME/.config/jarvis/google-oauth-client.json"
```

or both values supplied by that Desktop OAuth client:

```sh
export JARVIS_GMAIL_CLIENT_ID="..."
export JARVIS_GMAIL_CLIENT_SECRET="..."
```

Optional settings:

```sh
export JARVIS_GMAIL_TOKEN_FILE="data/gmail_oauth_token.json"
export JARVIS_GMAIL_QUERY="newer_than:30d"
```

The default token file is `data/gmail_oauth_token.json`, under the ignored local
data directory. Jarvis creates it with restrictive local file permissions and
uses it only to refresh OAuth authorization. Tokens are never written to
SQLite, external-source records, logs, prompts, or the web UI. Client secrets
and token file contents must remain local and must not be committed. Missing or
invalid OAuth configuration stops before the database is opened.

On first sync, Jarvis opens a loopback-only temporary callback, opens Google's
consent page in the local browser, checks the OAuth state value, and exchanges
the one-time code. Later runs refresh the access token locally. Callback
request logging is disabled so its one-time code is not logged. Google API
errors are reported without response bodies, message content, or credentials.

### Manual sync

From the project directory, after setting the local configuration above, run an
initial full sync (and later whenever recovery is needed):

```sh
./scripts/sync_gmail.sh
```

Then run incremental syncs explicitly:

```sh
./scripts/sync_gmail_incremental.sh
```

The incremental command requires a successful full sync for the same Gmail
account and configured query. Its checkpoint is scoped by a one-way hash of the
account identity and a hash of the exact normalized query; neither email address
nor query text is stored in the checkpoint. Changing accounts or query filters
therefore requires a full sync to initialize that scope. An expired Gmail
history checkpoint fails with an instruction to run full sync; it is never
silently treated as successful.

Optionally bound a full-sync invocation to a number of pages:

```sh
./scripts/sync_gmail.sh --max-pages 5
```

The full command fetches one page at a time and ingests through
`ExternalSourceRepository`. It captures a starting Gmail history ID, completes
the full query, catches up changes made during that scan, and only then saves
the resulting checkpoint. A page-limited/incomplete or failed scan does not
save a checkpoint. Incremental sync consumes all History pages, fetches each
changed message once, applies label/content changes through the same versioned
upsert, and soft-deletes missing or out-of-query messages from current search.
It advances the checkpoint only after every change succeeds. If interrupted,
the old checkpoint remains and the next run safely replays idempotent upserts
and tombstones. The optional `JARVIS_GMAIL_QUERY` narrows both full selection
and incremental membership checks. Commands print aggregate counts only; they
do not print account identity, query text, message content, or credentials.

### Local data and memory boundary

Jarvis keeps selected message ID, thread ID, subject, sender, recipients,
labels, timestamps, a safe Gmail reference, normalized readable body, content
hash/version, import time, and connector provenance in the local SQLite
external-source tables. MIME attachments and unrelated Gmail API fields are
not imported. The local `gmail_sync_state` table stores only hashed account and
query identities, the Gmail history ID, and update time. Deleted messages retain
their external-source record and versions with a deletion timestamp, while
their current evidence projection is excluded from search. The API's access and
refresh tokens, OAuth client secret, and authorization headers are never stored
in SQLite. Credentials are not passed to Ollama.

Ingested mail becomes searchable source evidence through existing FTS/recall.
Evidence text is untrusted data, not instructions; email content cannot change
Jarvis's system instructions. Gmail records do not create canonical memories
or automatically enter personal context. A future memory-candidate workflow
may propose an email-supported candidate, but it remains unconfirmed until the
user explicitly approves it.

There is no cloud database, third-party index, cloud LLM, Gmail UI, automatic
sync scheduler, or Google Pub/Sub integration.
