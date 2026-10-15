# Applying the canonical memory seed

Jarvis supports a curated, explicitly confirmed canonical memory seed: durable,
non-sensitive records (preferences, goals, projects, learnings) that are applied
through the normal canonical memory lifecycle (candidate creation plus explicit
confirmation), keeping provenance separate from ordinary citation evidence.

## Public vs. personal manifests

- `seeds/canonical_memories.example.json` — a synthetic example manifest that
  documents the schema. It contains **no** real user data.
- `data/seed/canonical_personal_memories.json` — where your own personal manifest
  lives. The `data/` directory is gitignored on purpose; personal seed records
  must never be committed.

`scripts/seed_canonical_memories.py --seed-file` defaults to the local manifest in
`data/seed/`. You can point it at another file, including the example, with
`--seed-file seeds/canonical_memories.example.json`.

## Applying

Review the manifest, then run this command from the repository root in the
Jarvis Python environment to apply it to the database selected by the normal
Jarvis configuration (`JARVIS_DB_PATH` or `config.yaml`):

```sh
./scripts/apply_canonical_memory_seed.sh
```

The command is idempotent. It skips a seed key already present or an equivalent
type-and-claim pair. It creates a concise provenance entry for each new record,
uses the canonical memory service to create a candidate, then explicitly
confirms that candidate with an audit reason. Seed records must be
non-sensitive; initial canonical seed content is validated before any database
writes.

Without `--apply`, the script does not open or change any database. Tests call
the seed service only with temporary databases; they never apply a manifest to
the local live database.