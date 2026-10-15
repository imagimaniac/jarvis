"""Jarvis CLI (Typer + Rich)."""
from __future__ import annotations
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from app.core import storage, extract, search as search_mod, utils
from app.core import config as config_mod

app = typer.Typer(name="jarvis", help="Jarvis - personal information + memory layer (local-first)")
console = Console()

def load_config():
    # Now resolved against the repo root rather than the working directory, so
    # the CLI and the server can no longer disagree about which file is in use.
    return config_mod.load_config()

def get_paths(cfg: dict):
    raw = config_mod.raw_dir()
    dbp = config_mod.db_path()
    raw.mkdir(parents=True, exist_ok=True)
    dbp.parent.mkdir(parents=True, exist_ok=True)
    return raw, dbp

@app.command("add")
def add_cmd(text: Optional[str] = typer.Argument(None, help="Text to capture; if omitted, read stdin"), source: str = typer.Option("cli"), origin_kind: str = typer.Option("self")):
    if text is None:
        try:
            text = sys.stdin.read().rstrip('\n\r')
        except Exception:
            text = ""
    if not text.strip():
        console.print("[yellow]Nothing to add[/yellow]")
        raise typer.Exit(0)
    cfg = load_config()
    raw_dir, dbp = get_paths(cfg)
    db = storage.JarvisDB(dbp)
    dt = utils.utc_now()
    fpath, relpath, ch = utils.write_raw(raw_dir, text, dt)
    eid = db.insert_entry(
        text=text,
        source=source,
        origin_kind=origin_kind,
        source_type="text",
        raw_path=fpath,
        raw_relpath=relpath,
        content_hash=ch,
    )
    ex = extract.HeuristicExtractor().extract(text, utils.utc_iso(dt))
    db.add_structured(eid, ex, inferred_by="heuristic")
    db.close()
    console.print(f"[green]Added[/green] #{eid}  [dim]{relpath}[/dim]  [dim]{source}/{origin_kind}[/dim]")

@app.command("search")
def search_cmd(query: str, limit: int = typer.Option(10, min=1, max=100), offset: int = typer.Option(0, min=0)):
    cfg = load_config()
    raw_dir, dbp = get_paths(cfg)
    db = storage.JarvisDB(dbp)
    try:
        l = int(limit)
        o = int(offset)
    except Exception:
        l, o = 10, 0
    res = db.search_fts(query, limit=l+o)
    res = search_mod.rank_results(res, q=query, self_boost=cfg.get('search', {}).get('self_boost', 0.25), recency_boost=cfg.get('search', {}).get('recency_boost', 0.15))
    res = res[o:o+l] if o or l else res
    table = Table(title=f"Search: {query}")
    table.add_column("ID")
    table.add_column("Date (UTC)")
    table.add_column("Src")
    table.add_column("Kind")
    table.add_column("Snippet")
    table.add_column("Source")
    for r in res:
        sn = r.get('snippet') or ''
        table.add_row(str(r['id']), (r['created_at_utc'] or '')[:19], r['source'] or '', r['origin_kind'] or '', sn, r.get('raw_relpath') or '')
    console.print(table)
    db.close()

@app.command("ask")
def ask_cmd(query: str, limit: int = typer.Option(10, min=1, max=100), offset: int = typer.Option(0, min=0)):
    cfg = load_config()
    syn_on = cfg.get('llm', {}).get('retrieval_synthesis', {}).get('enabled', False) or cfg.get('features', {}).get('reflection', False)
    if syn_on:
        console.print("[dim]Synthesis enabled (labeled SYNTHESIZED) — showing sources[/dim]")
    search_cmd(query=query, limit=limit, offset=offset)
    console.print("[dim](search-first; synthesis only if explicitly enabled)[/dim]")

@app.command("show")
def show_cmd(eid: int, json_out: bool = typer.Option(False, "--json")):
    cfg = load_config()
    raw_dir, dbp = get_paths(cfg)
    db = storage.JarvisDB(dbp)
    g = db.get(eid)
    if not g or not g['entry']:
        console.print("[red]Not found[/red]")
        raise typer.Exit(1)
    e = g['entry']
    s = g['structured']
    if json_out:
        import json as _json
        out = {"entry": e, "structured": s, "relations": g['relations']}
        console.print(_json.dumps(out, ensure_ascii=False, default=str))
    else:
        console.rule(f"#{e['id']} {(e['created_at_utc'] or '')[:19]}Z  {e['source']}/{e['origin_kind']}")
        console.print(e['text_raw'])
        if s:
            console.rule("Structured")
            console.print(s)
        if g['relations']:
            console.rule("Relations")
            console.print(g['relations'])
        console.rule("Source")
        console.print(f"raw_path: {e['raw_path']}")
        console.print(f"raw_relpath: {e['raw_relpath']}")
        console.print(f"content_hash: {e['content_hash']}")
    db.close()

@app.command("recent")
def recent_cmd(n: int = typer.Option(20, min=1, max=100), offset: int = typer.Option(0, min=0)):
    cfg = load_config()
    raw_dir, dbp = get_paths(cfg)
    db = storage.JarvisDB(dbp)
    try:
        l = int(n)
        o = int(offset)
    except Exception:
        l, o = 20, 0
    rows = db.recent(limit=l+o)
    rows = rows[o:o+l] if o or l else rows
    table = Table(title="Recent")
    table.add_column("ID")
    table.add_column("Date (UTC)")
    table.add_column("Src")
    table.add_column("Kind")
    table.add_column("Snippet")
    table.add_column("RelPath")
    for r in rows:
        table.add_row(str(r['id']), (r['created_at_utc'] or '')[:19], r['source'] or '', r['origin_kind'] or '', r.get('snippet') or '', r.get('raw_relpath') or '')
    console.print(table)
    db.close()

@app.command("stats")
def stats_cmd(json_out: bool = typer.Option(False, "--json")):
    cfg = load_config()
    raw_dir, dbp = get_paths(cfg)
    db = storage.JarvisDB(dbp)
    st = db.stats()
    if json_out:
        import json as _json
        console.print(_json.dumps(st, ensure_ascii=False))
    else:
        table = Table(title="Stats")
        table.add_column("Key")
        table.add_column("Value")
        for k,v in st.items():
            table.add_row(k, str(v))
        console.print(table)
    db.close()

@app.command("missed")
def missed_cmd(limit: int = typer.Option(5)):
    cfg = load_config()
    rem = cfg.get('reminders', {}).get('enabled', False)
    if not rem:
        console.print("[dim]Reminders are disabled (reminders.enabled=false). Nothing to show.[/dim]")
        console.print("[dim]Deterministic 'you missed this' is planned for Phase 2.[/dim]")
        return
    console.print("[dim]'You missed this' is not implemented yet (Phase 2).[/dim]")

@app.command("version")
def version_cmd():
    console.print("jarvis 0.1.1 (polish+utilities)")

if __name__ == "__main__":
    app()
