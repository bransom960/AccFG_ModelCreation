#!/usr/bin/env python3
"""Stage 1: sample (ID, SMILES) pairs from a directory of .jsonl files into flat text shards.

Each output line is `<cid>\t<smiles>`, one .smi shard per input file. The ID is what lets
every later stage say which molecule ended up in which cluster and model.

A record is ELIGIBLE when it has a SMILES and an ID and is not a salt or mixture (its SMILES
contains no '.'). Records that are not eligible are counted and skipped, never sampled.

Two sampling modes:

  --target N  (default, N = 500,000)
      Exactly N eligible molecules, drawn uniformly at random from ALL eligible records in
      ALL files: every eligible molecule has the same chance, however the files are sized.
      Pass 1 counts the eligible records in each file (cached in _eligible_counts.json).
      The per-file quotas are then one multivariate-hypergeometric draw, which is exactly
      what a uniform N-of-total sample implies. Pass 2 picks each file's quota at random
      positions spread over the whole file. A picked SMILES that RDKit cannot parse is
      replaced by another random eligible record from the same file, so the shards hold N
      parseable molecules (unless the files hold fewer eligible records than N, which is an
      error).

  --fraction F
      The older mode: a random fraction F of the lines of EVERY file, salts still dropped.
      The total is whatever F of the lines turns out to be.

Sampling is seeded (--seed), so the same command selects the same molecules regardless of
worker count or completion order. One worker process per input file; workers write their
own output file and return only counts.

    python extract_smiles.py /data/jsonl -o /data/smiles                    # 500k molecules
    python extract_smiles.py /data/jsonl -o /data/smiles --target 1000000   # 1M molecules
    python extract_smiles.py /data/jsonl -o /data/smiles --fraction 0.03    # 3% of each file
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Matches "<field>": "<value>" capturing the raw (still JSON-escaped) value.
# (?:[^"\\]|\\.)* correctly walks over \" and \\ so SMILES like C/C=C\C don't truncate.
_CACHE: dict[str, re.Pattern[bytes]] = {}

DEFAULT_TARGET = 500_000


def _pattern(field: str) -> re.Pattern[bytes]:
    if field not in _CACHE:
        _CACHE[field] = re.compile(
            rb'"' + re.escape(field.encode()) + rb'"\s*:\s*"((?:[^"\\]|\\.)*)"'
        )
    return _CACHE[field]


def _id_pattern(field: str) -> re.Pattern[bytes]:
    """Like _pattern, but also accepts a bare JSON number ("cid": 123 as well as "cid": "123")."""
    key = "#id:" + field
    if key not in _CACHE:
        _CACHE[key] = re.compile(
            rb'"' + re.escape(field.encode()) + rb'"\s*:\s*(?:"((?:[^"\\]|\\.)*)"|(-?\d+))'
        )
    return _CACHE[key]


def _unescape(raw: bytes) -> str:
    """Decode a captured JSON string body. Fast path for the common no-escape case."""
    if b"\\" not in raw:
        return raw.decode("utf-8")
    return json.loads(b'"' + raw + b'"')


def default_jobs() -> int:
    """Cores this job may use -- NOT os.cpu_count(), which on a cluster node reports the
    whole machine. Under LSF, the slots `bsub -n` granted on this host (LSB_MCPU_HOSTS);
    otherwise the CPU affinity mask, which respects cgroups and taskset.
    (Duplicated in fg_matrix.py so each script stays standalone.)"""
    try:
        avail = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        avail = os.cpu_count() or 1
    tokens = os.environ.get("LSB_MCPU_HOSTS", "").split()  # "hostA 8 hostB 8"
    if tokens:
        slots = {h.split(".")[0]: int(n) for h, n in zip(tokens[0::2], tokens[1::2])}
        here = socket.gethostname().split(".")[0]
        if len(slots) > 1:
            print(f"WARNING: LSF spread this job over {len(slots)} hosts {slots}. Worker "
                  f"processes can only use this host ({here}); request "
                  f'-R "span[hosts=1]" to get all slots on one node.',
                  file=sys.stderr, flush=True)
        if here in slots:
            return max(1, min(slots[here], avail))
    return max(1, avail)


def _file_rng(seed: int, name: str) -> random.Random:
    """Per-file RNG. crc32, not hash(): str hashing is salted per process, which would
    make the sample change between runs."""
    return random.Random(seed * 1_000_003 + zlib.crc32(name.encode()))


def parse_record(line: bytes, pat, id_pat) -> tuple[str, str, str]:
    """Return (status, cid, smiles) for one JSONL line.

    status is 'ok' (eligible), 'no_smiles', 'no_id' or 'salt' (SMILES contains '.', i.e.
    more than one disconnected component: a salt, solvate or mixture).
    """
    m = pat.search(line)
    smi = _unescape(m.group(1)) if m is not None else ""
    if not smi:
        return "no_smiles", "", ""
    m_id = id_pat.search(line)
    mol_id = ""
    if m_id is not None:
        mol_id = (_unescape(m_id.group(1)) if m_id.group(1) is not None
                  else m_id.group(2).decode()).strip()
    if not mol_id or "\t" in mol_id:
        return "no_id", "", smi
    if "." in smi:
        return "salt", mol_id, smi
    return "ok", mol_id, smi


def iter_eligible(src: Path, field: str, id_field: str, stats: dict):
    """Yield (cid, smiles) for each eligible record, in file order; tally the rest in stats."""
    pat, id_pat = _pattern(field), _id_pattern(id_field)
    with src.open("rb") as fh:
        for line in fh:
            stats["lines"] += 1
            status, mol_id, smi = parse_record(line, pat, id_pat)
            stats[status] += 1
            if status == "ok":
                yield mol_id, smi


def _new_stats() -> dict:
    return {"lines": 0, "ok": 0, "no_smiles": 0, "no_id": 0, "salt": 0}


# ---- --target mode ------------------------------------------------------------------

def count_one(args: tuple[str, str, str]) -> dict:
    """Pass 1: count the eligible records in one file. Runs in a worker process."""
    src_s, field, id_field = args
    src = Path(src_s)
    stats = _new_stats()
    for _ in iter_eligible(src, field, id_field, stats):
        pass
    return {"src": src.name, **stats}


def eligible_counts(files: list[Path], field: str, id_field: str, cache_file: Path,
                    jobs: int) -> dict[str, dict]:
    """Per-file record counts, cached. Keyed on (size, mtime) so a changed file is recounted."""
    cache: dict = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            cache = {}

    def key(f: Path) -> list:
        st = f.stat()
        return [st.st_size, int(st.st_mtime)]

    stale = [f for f in files if cache.get(f.name, {}).get("key") != key(f)]
    if stale:
        print(f"pass 1: counting eligible records in {len(stale)} file(s) on {jobs} workers ...",
              flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(count_one, (str(f), field, id_field)): f for f in stale}
            for i, fut in enumerate(as_completed(futs), 1):
                f = futs[fut]
                r = fut.result()
                cache[f.name] = {"key": key(f), **{k: r[k] for k in _new_stats()}}
                if i % 25 == 0 or i == len(stale):
                    print(f"  counted {i}/{len(stale)} files "
                          f"({time.perf_counter() - t0:,.0f}s)", flush=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    else:
        print("pass 1: eligible-record counts cached", flush=True)
    return {f.name: cache[f.name] for f in files}


def allocate(eligible: list[int], target: int, seed: int) -> list[int]:
    """Per-file quotas for a uniform sample of `target` records from all eligible records.

    Drawing `target` records uniformly without replacement from the pooled records makes
    each file's share multivariate-hypergeometric; drawing the quotas that way and then
    sampling uniformly inside each file is the same as sampling the pool directly.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    return [int(n) for n in rng.multivariate_hypergeometric(
        np.asarray(eligible, dtype=np.int64), target)]


def _parses(smiles: str) -> bool:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles) is not None


def sample_one(args: tuple[str, str, str, str, int, int, int, bool]) -> dict:
    """Pass 2: write `quota` eligible records of one file, chosen at random. Runs in a
    worker process.

    A random permutation of the file's eligible-record positions fixes the order in which
    records are tried: the first `quota` are the sample, the rest are replacements, used
    in order, for picks whose SMILES RDKit cannot parse.
    """
    src_s, out_s, field, id_field, quota, n_eligible, seed, check_parse = args
    src, out = Path(src_s), Path(out_s)
    t0 = time.perf_counter()
    order = list(range(n_eligible))
    _file_rng(seed, src.name).shuffle(order)

    chosen: dict[int, tuple[str, str]] = {}
    unparseable = 0
    tried = 0          # how many positions of `order` have been decided
    want = quota
    while len(chosen) < quota and tried < n_eligible:
        # Read the next block of candidate positions in one pass over the file.
        block = order[tried:min(n_eligible, tried + want + max(20, want // 50))]
        wanted = set(block)
        found: dict[int, tuple[str, str]] = {}
        stats = _new_stats()
        for idx, (mol_id, smi) in enumerate(iter_eligible(src, field, id_field, stats)):
            if idx in wanted:
                found[idx] = (mol_id, smi)
        if stats["ok"] != n_eligible:
            raise RuntimeError(f"{src.name} has {stats['ok']} eligible records now, "
                               f"{n_eligible} when counted; delete _eligible_counts.json")
        for idx in block:
            tried += 1
            mol_id, smi = found[idx]
            if check_parse and not _parses(smi):
                unparseable += 1
                continue
            chosen[idx] = (mol_id, smi)
            if len(chosen) == quota:
                break
        want = quota - len(chosen)

    tmp = out.with_suffix(out.suffix + ".partial")
    with tmp.open("w", encoding="utf-8", newline="\n") as w:
        for idx in sorted(chosen):          # file order
            mol_id, smi = chosen[idx]
            w.write(f"{mol_id}\t{smi}\n")
    os.replace(tmp, out)  # atomic: a complete .smi on disk means that file is done
    return {"src": src.name, "written": len(chosen), "quota": quota,
            "unparseable": unparseable, "secs": round(time.perf_counter() - t0, 2)}


# ---- --fraction mode ----------------------------------------------------------------

def extract_one(args: tuple[str, str, str, str, bool, float, int]) -> dict:
    """Write a random `fraction` of one file's lines (eligible ones only). Worker process."""
    src_s, out_s, field, id_field, dedupe, fraction, seed = args
    src, out = Path(src_s), Path(out_s)
    pat = _pattern(field)
    id_pat = _id_pattern(id_field)
    draw = _file_rng(seed, src.name).random
    take_all = fraction >= 1.0

    tmp = out.with_suffix(out.suffix + ".partial")
    stats = _new_stats()
    n_sampled = n_dup = 0
    seen: set[str] | None = set() if dedupe else None
    t0 = time.perf_counter()

    with src.open("rb") as fh, tmp.open("w", encoding="utf-8", newline="\n") as w:
        write = w.write
        for line in fh:
            stats["lines"] += 1
            # Draw BEFORE parsing, so the skipped lines cost almost nothing.
            if not take_all and draw() >= fraction:
                continue
            n_sampled += 1
            status, mol_id, smi = parse_record(line, pat, id_pat)
            stats[status] += 1
            if status != "ok":
                continue
            if seen is not None:
                if smi in seen:
                    n_dup += 1
                    continue
                seen.add(smi)
            write(mol_id)
            write("\t")
            write(smi)
            write("\n")

    os.replace(tmp, out)
    return {"src": src.name, "lines": stats["lines"], "sampled": n_sampled,
            "written": stats["ok"] - n_dup, "no_smiles": stats["no_smiles"],
            "no_id": stats["no_id"], "salt": stats["salt"], "dupes": n_dup,
            "secs": round(time.perf_counter() - t0, 2)}


# ---- driver ---------------------------------------------------------------------------

def check_params(outdir: Path, params: dict) -> bool:
    """Record the sampling parameters; refuse to resume into a directory made with others."""
    params_file = outdir / "_params.json"
    if params_file.exists():
        prev = json.loads(params_file.read_text())
        if prev != params:
            print(f"ERROR: {params_file} was written with {prev},\n"
                  f"       but this run uses {params}. Use a fresh --outdir.", file=sys.stderr)
            return False
    else:
        params_file.write_text(json.dumps(params, indent=1))
    return True


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as fh:
        return sum(1 for _ in fh)


def run_target(args, files: list[Path]) -> int:
    if not args.no_parse_check:
        try:
            import rdkit  # noqa: F401  (workers import it; fail here, clearly, if it is missing)
        except ImportError:
            print("ERROR: RDKit is needed to replace unparseable picks. Install it, or pass "
                  "--no-parse-check.", file=sys.stderr)
            return 1
    counts = eligible_counts(files, args.field, args.id_field,
                             args.outdir / "_eligible_counts.json", args.jobs)
    eligible = [counts[f.name]["ok"] for f in files]
    total = sum(eligible)
    lines = sum(c["lines"] for c in counts.values())
    salts = sum(c["salt"] for c in counts.values())
    print(f"  {lines:,} records in {len(files)} files: {total:,} eligible, {salts:,} salts/"
          f"mixtures, {sum(c['no_id'] for c in counts.values()):,} without "
          f"'{args.id_field}', {sum(c['no_smiles'] for c in counts.values()):,} without "
          f"'{args.field}'", flush=True)
    if total < args.target:
        print(f"ERROR: only {total:,} eligible records, fewer than --target {args.target:,}.",
              file=sys.stderr)
        return 1

    quotas = allocate(eligible, args.target, args.seed)
    (args.outdir / "_quotas.json").write_text(json.dumps(
        {f.name: q for f, q in zip(files, quotas)}, indent=1))

    tasks = []
    skipped = 0
    for f, q, n in zip(files, quotas, eligible):
        out = args.outdir / (f.stem + ".smi")
        if args.resume and out.exists():
            skipped += 1
            continue
        tasks.append((str(f), str(out), args.field, args.id_field, q, n, args.seed,
                      not args.no_parse_check))
    print(f"pass 2: sampling {args.target:,} molecules ({100 * args.target / total:.2f}% of "
          f"eligible); {skipped} file(s) already done, {len(tasks)} to write", flush=True)

    written = unparseable = short = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(sample_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:
                print(f"  !! FAILED {Path(futs[fut]).name}: {e!r}", file=sys.stderr, flush=True)
                short += 1
                continue
            written += r["written"]
            unparseable += r["unparseable"]
            if r["written"] < r["quota"]:
                short += 1
                print(f"  !! {r['src']}: only {r['written']} of {r['quota']} parse",
                      file=sys.stderr, flush=True)
            if i % 25 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {written:,} written "
                      f"({time.perf_counter() - t0:,.0f}s)", flush=True)

    on_disk = sum(_count_lines(args.outdir / (f.stem + ".smi")) for f in files)
    print(f"\ndone: {on_disk:,} molecules in {args.outdir} (target {args.target:,}); "
          f"{unparseable:,} unparseable picks replaced")
    if on_disk != args.target or short:
        print("ERROR: the shards do not hold exactly the target number of molecules; see "
              "the messages above.", file=sys.stderr)
        return 1
    return 0


def run_fraction(args, files: list[Path]) -> int:
    tasks = []
    skipped = 0
    for f in files:
        out = args.outdir / (f.stem + ".smi")
        if args.resume and out.exists():
            skipped += 1
            continue
        tasks.append((str(f), str(out), args.field, args.id_field, args.dedupe,
                      args.fraction, args.seed))
    print(f"{len(files)} input files, {skipped} already done, {len(tasks)} to process, "
          f"{args.jobs} workers, fraction={args.fraction:g}, seed={args.seed}", flush=True)
    if not tasks:
        return 0

    tot = {"lines": 0, "sampled": 0, "written": 0, "no_smiles": 0, "no_id": 0, "salt": 0,
           "dupes": 0}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(extract_one, t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:  # keep going; one bad file shouldn't sink the run
                print(f"  !! FAILED {Path(futs[fut]).name}: {e!r}", file=sys.stderr, flush=True)
                continue
            for k in tot:
                tot[k] += r[k]
            print(f"[{i}/{len(tasks)}] {r['src']:<32} {r['written']:>8,} of {r['lines']:>8,} "
                  f"lines ({r['salt']:,} salts, {r['no_id']:,} no id, {r['dupes']:,} dup) "
                  f"{r['secs']:>6.1f}s | total so far {tot['written']:,}", flush=True)

    el = time.perf_counter() - t0
    print(f"\ndone in {el/60:.1f} min")
    print(f"  lines read     {tot['lines']:,}")
    print(f"  lines sampled  {tot['sampled']:,}")
    print(f"  smiles written {tot['written']:,}")
    print(f"  salts/mixtures {tot['salt']:,}  (dropped)")
    print(f"  field missing  {tot['no_smiles']:,}")
    print(f"  no ID          {tot['no_id']:,}  (no '{args.id_field}' field; skipped)")
    if args.dedupe:
        print(f"  in-shard dupes {tot['dupes']:,}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indir", type=Path, help="directory containing .jsonl files")
    ap.add_argument("-o", "--outdir", type=Path, required=True,
                    help="directory for .smi shards")
    ap.add_argument("--field", default="smiles",
                    help="JSON key to pull. Default 'smiles' = PubChem Isomeric SMILES, which "
                         "KEEPS stereochemistry. 'c-smiles' is PubChem Canonical SMILES and has "
                         "stereo STRIPPED -- measured 0 specified stereo elements across a real "
                         "sample vs 32 in `smiles`. Only use c-smiles if you want stereo gone.")
    ap.add_argument("--id-field", default="cid",
                    help="JSON key holding the molecule ID (default 'cid'). Written as the "
                         "first column of every shard line and carried through every later stage")
    ap.add_argument("-j", "--jobs", type=int, default=None,
                    help="worker processes (default: cores allocated to this job -- under "
                         "LSF, the slots from bsub -n on this host)")
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--target", type=int, default=None,
                    help=f"sample exactly this many eligible molecules across all files "
                         f"(default {DEFAULT_TARGET:,} unless --fraction is given)")
    ap.add_argument("--fraction", type=float, default=None,
                    help="instead of --target: sample this fraction of the lines of EACH file "
                         "(1 = every eligible line)")
    ap.add_argument("--no-parse-check", action="store_true",
                    help="--target mode: do not replace picks that RDKit cannot parse "
                         "(skips the RDKit import; fg_matrix.py then rejects them instead)")
    ap.add_argument("--dedupe", action="store_true",
                    help="--fraction mode only: drop duplicate SMILES within each shard, keeping "
                         "the first ID seen")
    ap.add_argument("--resume", action="store_true",
                    help="skip inputs whose .smi shard already exists")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed (default 0). Same seed -> same molecules selected")
    args = ap.parse_args()
    if args.jobs is None:
        args.jobs = default_jobs()

    if args.fraction is not None and args.target is not None:
        print("give --target or --fraction, not both", file=sys.stderr)
        return 1
    if args.fraction is None and args.target is None:
        args.target = DEFAULT_TARGET
    if args.fraction is not None and not 0 < args.fraction <= 1:
        print("--fraction must be in (0, 1]", file=sys.stderr)
        return 1
    if args.target is not None and (args.target <= 0 or args.dedupe):
        print("--target must be positive, and --dedupe is only available with --fraction",
              file=sys.stderr)
        return 1

    files = sorted(args.indir.glob(args.glob))
    if not files:
        print(f"no files matching {args.glob} in {args.indir}", file=sys.stderr)
        return 1
    args.outdir.mkdir(parents=True, exist_ok=True)

    params = {"field": args.field, "id_field": args.id_field, "glob": args.glob,
              "target": args.target, "fraction": args.fraction, "seed": args.seed,
              "dedupe": args.dedupe, "drop_salts": True,
              "parse_check": args.target is not None and not args.no_parse_check}
    if not check_params(args.outdir, params):
        return 2

    if args.field == "c-smiles":
        print("WARNING: 'c-smiles' is PubChem Canonical SMILES -- stereochemistry is stripped.\n"
              "         Use --field smiles unless you specifically want stereo discarded.",
              file=sys.stderr, flush=True)
    print(f"{len(files)} input files matching {args.glob!r}, {args.jobs} workers, "
          f"field={args.field!r}, seed={args.seed}", flush=True)

    t0 = time.perf_counter()
    rc = run_target(args, files) if args.target is not None else run_fraction(args, files)
    print(f"total time {(time.perf_counter() - t0) / 60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
