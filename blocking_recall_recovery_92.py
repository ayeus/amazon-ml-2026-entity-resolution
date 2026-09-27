"""High-recall candidate recovery for the Amazon ML entity-resolution data.

Run in Colab after mounting Google Drive:
    %run /content/blocking_recall_recovery_92.py

The script reuses saved name-token candidates, then adds bounded address/name
token and number blocks. It evaluates pair recall against the supplied labels,
and writes a final deduplicated candidate Parquet for each labeled split.

Important: the script does not claim 92% unless the measured VALIDATION recall
reaches that target. If it does not, it leaves a report explaining the result.
Blocking recall depends on the actual data; no code can guarantee a target
before running against those files.
"""

from pathlib import Path
import json
import shutil
import time

try:
    from google.colab import drive
    if not Path('/content/drive/MyDrive').exists():
        drive.mount('/content/drive')
except ImportError:
    pass  # Also usable outside Colab when ROOT is edited below.

import duckdb


# ------------------------------- Configuration ------------------------------
ROOT = Path('/content/drive/MyDrive/Amazon_ML_Challenge_2026')
DATA = ROOT / 'entity_resolution_processed'
BLOCKING = ROOT / 'entity_resolution_blocking'
EXPERIMENT = BLOCKING / 'recall_experiments' / 'v2_recall92'
if not ROOT.is_dir():
    raise FileNotFoundError(
        f'Google Drive project folder not found: {ROOT}. '
        'Mount Drive in Colab and confirm the folder is named '
        'MyDrive/Amazon_ML_Challenge_2026, or edit ROOT near the top of this script.'
    )
if not DATA.is_dir():
    raise FileNotFoundError(f'Processed-data folder not found: {DATA}')
if not BLOCKING.is_dir():
    raise FileNotFoundError(f'Blocking-output folder not found: {BLOCKING}')
EXPERIMENT.mkdir(parents=True, exist_ok=True)

SPLIT_FILE = DATA / 'train_source1_split.tsv'
POSITIVE_FILE = DATA / 'train_positive_pairs.tsv'
SOURCE_FILES = {
    's1': DATA / 'train_source1.parquet',
    's2': DATA / 'train_source2.parquet',
    's3': DATA / 'train_source3.parquet',
}
BASELINE = {
    'train': BLOCKING / 'rule_candidates' / 'union_train.parquet',
    'validation': BLOCKING / 'validation' / 'union_validation.parquet',
}

# Reuse these exact saved artifacts from the earlier name-token run when present.
SAVED_NAME_TOKENS = {
    'train': BLOCKING / 'recall_experiments' / 'v1' / 'token_train_business_name_tokens.parquet',
    'validation': BLOCKING / 'recall_experiments' / 'v1' / 'token_validation_business_name_tokens.parquet',
}

# Each field is evaluated with increasingly permissive frequency caps. The
# script selects the most permissive tier whose estimated pair work fits the
# per-field and per-split budgets. These are upper bounds before pair dedup.
CAP_TIERS = [
    (500, 5_000),
    (1_000, 10_000),
    (2_500, 25_000),
    (5_000, 50_000),
    (10_000, 100_000),
]
MAX_ESTIMATED_PAIRS_PER_FIELD = 220_000_000
MAX_ESTIMATED_PAIRS_PER_SPLIT = 600_000_000
MAX_CANDIDATE_OUTPUT_GB = 55
DUCKDB_MEMORY = '2GB'  # low enough for Colab; work spills to local /content disk
PARTITIONS = 128       # disk-backed deduplication; increase only if a partition OOMs
TARGET_RECALL = 0.92

LOCAL = Path('/content/recall92_work')
LOCAL.mkdir(parents=True, exist_ok=True)
DUCK_TMP = LOCAL / 'duckdb_tmp'
DUCK_TMP.mkdir(parents=True, exist_ok=True)


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def required_file(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Required input is missing or empty: {path}')


for _p in [SPLIT_FILE, POSITIVE_FILE, *SOURCE_FILES.values()]:
    required_file(_p)

if shutil.disk_usage('/content').free / (1024 ** 3) < 12:
    raise RuntimeError('Need at least 12 GB free on Colab local /content disk for spill files.')

con = duckdb.connect(':memory:', config={
    'threads': '2',
    'memory_limit': DUCKDB_MEMORY,
    'temp_directory': str(DUCK_TMP),
    'max_temp_directory_size': '65GB',
    'preserve_insertion_order': 'false',
})
con.execute("SET enable_progress_bar=true")
print('Experiment output:', EXPERIMENT)
print('Local free disk GB:', round(shutil.disk_usage('/content').free / (1024 ** 3), 1))
print('DuckDB memory:', DUCKDB_MEMORY, '| partition count:', PARTITIONS)

SOURCE_COLUMNS = {}
SOURCE_SCHEMA = {}
for source_key, source_path in SOURCE_FILES.items():
    _schema = con.execute(
        f'DESCRIBE SELECT * FROM read_parquet({sql_path(source_path)})'
    ).fetchdf()
    SOURCE_COLUMNS[source_key] = set(_schema['column_name'].tolist())
    SOURCE_SCHEMA[source_key] = _schema


# ---------------------------- Input / baseline checks ------------------------
for split, p in BASELINE.items():
    if p.is_file() and p.stat().st_size:
        try:
            con.execute(f'SELECT 1 FROM read_parquet({sql_path(p)}) LIMIT 1').fetchone()
            print('Baseline usable:', split, p, f'({p.stat().st_size:,} bytes)')
        except Exception as exc:
            print('Baseline ignored (unreadable):', split, p, str(exc)[:240])
    else:
        print('Baseline ignored (missing/empty):', split, p)


def existing_candidate_paths(split):
    paths = []
    base = BASELINE[split]
    if base.is_file() and base.stat().st_size:
        try:
            con.execute(f'SELECT 1 FROM read_parquet({sql_path(base)}) LIMIT 1').fetchone()
            paths.append((base, 'baseline'))
        except Exception:
            pass
    name_path = SAVED_NAME_TOKENS[split]
    if name_path.is_file() and name_path.stat().st_size:
        con.execute(f'SELECT 1 FROM read_parquet({sql_path(name_path)}) LIMIT 1').fetchone()
        paths.append((name_path, 'name_tokens_saved'))
    return paths


def recall_metrics(split, path_tags):
    """Count distinct supplied positives covered; candidate duplicates do not inflate recall."""
    split_value = 'train' if split == 'train' else 'validation'
    if not path_tags:
        raise RuntimeError(f'No candidate sources available for {split}.')
    candidate_sql = ' UNION ALL '.join(
        f"SELECT CAST(source1_entity_id AS VARCHAR) s1, "
        f"CAST(candidate_entity_id AS VARCHAR) cand FROM read_parquet({sql_path(p)})"
        for p, _tag in path_tags
    )
    row_upper = sum(int(con.execute(
        f'SELECT count(*) FROM read_parquet({sql_path(p)})'
    ).fetchone()[0]) for p, _tag in path_tags)
    result = con.execute(f"""
        WITH truth AS (
            SELECT DISTINCT p.source1_entity_id, p.matched_entity_id
            FROM read_csv({sql_path(POSITIVE_FILE)}, delim='\\t', header=true, all_varchar=true) p
            JOIN read_csv({sql_path(SPLIT_FILE)}, delim='\\t', header=true, all_varchar=true) s
              ON p.source1_entity_id=s.source1_entity_id
            WHERE s.split='{split_value}'
        ),
        covered AS (
            SELECT DISTINCT t.source1_entity_id, t.matched_entity_id
            FROM truth t SEMI JOIN ({candidate_sql}) c
              ON t.source1_entity_id=c.s1 AND t.matched_entity_id=c.cand
        )
        SELECT (SELECT count(*) FROM truth), (SELECT count(*) FROM covered)
    """).fetchone()
    truth_n, hit_n = map(int, result)
    return {
        'split': split,
        'truth_pairs': truth_n,
        'captured_positive_pairs': hit_n,
        'blocking_recall': hit_n / truth_n if truth_n else None,
        'candidate_rows_upper_bound': row_upper,
    }


# -------------------------- Bounded token block builder ----------------------
def build_token_block(split, token_column, remaining_budget, fixed_caps=None):
    """Build one shared-token block after profiling keys; skip safely if too large."""
    allowed = {
        'business_name_tokens',
        'business_address_tokens',
        'business_name_numbers',
        'business_address_numbers',
    }
    if token_column not in allowed:
        raise ValueError(f'Unsupported token column: {token_column}')
    missing = [key for key in SOURCE_FILES if token_column not in SOURCE_COLUMNS[key]]
    if missing:
        print(f'SKIP {split} {token_column}: column absent from {missing}')
        return None, 0, fixed_caps

    # Drop this field's old temp tables; the connection is created before any work.
    for suffix in ['s1', 'target', 's1df', 'targetdf', 'allowed']:
        con.execute(f'DROP TABLE IF EXISTS {suffix}_{split}_{token_column}')
    s1_table = f's1_{split}_{token_column}'
    target_table = f'target_{split}_{token_column}'
    s1df_table = f's1df_{split}_{token_column}'
    tdf_table = f'targetdf_{split}_{token_column}'
    allowed_table = f'allowed_{split}_{token_column}'
    col = '"' + token_column.replace('"', '""') + '"'
    split_name = 'train' if split == 'train' else 'validation'

    con.execute(f"""
        CREATE TEMP TABLE {s1_table} AS
        SELECT DISTINCT CAST(a.entity_id AS VARCHAR) AS source1_entity_id,
               lower(trim(CAST(u.token AS VARCHAR))) AS token
        FROM read_parquet({sql_path(SOURCE_FILES['s1'])}) a
        JOIN read_csv({sql_path(SPLIT_FILE)}, delim='\\t', header=true, all_varchar=true) s
          ON CAST(a.entity_id AS VARCHAR)=s.source1_entity_id
        CROSS JOIN UNNEST(a.{col}) AS u(token)
        WHERE s.split='{split_name}' AND u.token IS NOT NULL
          AND length(trim(CAST(u.token AS VARCHAR))) >= 2
    """)
    con.execute(f"""
        CREATE TEMP TABLE {target_table} AS
        SELECT DISTINCT CAST(b.entity_id AS VARCHAR) AS candidate_entity_id,
               'S2' AS candidate_source, lower(trim(CAST(u.token AS VARCHAR))) AS token
        FROM read_parquet({sql_path(SOURCE_FILES['s2'])}) b
        CROSS JOIN UNNEST(b.{col}) AS u(token)
        WHERE u.token IS NOT NULL AND length(trim(CAST(u.token AS VARCHAR))) >= 2
        UNION ALL
        SELECT DISTINCT CAST(b.entity_id AS VARCHAR) AS candidate_entity_id,
               'S3' AS candidate_source, lower(trim(CAST(u.token AS VARCHAR))) AS token
        FROM read_parquet({sql_path(SOURCE_FILES['s3'])}) b
        CROSS JOIN UNNEST(b.{col}) AS u(token)
        WHERE u.token IS NOT NULL AND length(trim(CAST(u.token AS VARCHAR))) >= 2
    """)
    con.execute(f'CREATE TEMP TABLE {s1df_table} AS SELECT token, count(*) df FROM {s1_table} GROUP BY token')
    con.execute(f'CREATE TEMP TABLE {tdf_table} AS SELECT candidate_source, token, count(*) df FROM {target_table} GROUP BY 1,2')

    chosen = None
    if fixed_caps is not None:
        max_df, max_product = fixed_caps
        estimated = int(con.execute(f"""
            SELECT coalesce(sum(s.df*t.df),0)
            FROM {s1df_table} s JOIN {tdf_table} t USING(token)
            WHERE s.df <= {max_df} AND t.df <= {max_df}
              AND s.df*t.df <= {max_product}
        """).fetchone()[0])
        fits = estimated <= MAX_ESTIMATED_PAIRS_PER_FIELD and estimated <= remaining_budget
        print(f'{split} {token_column}: reusing train-selected caps df={max_df:,}, '
              f'product={max_product:,}; estimate={estimated:,}; fits={fits}')
        if fits and estimated > 0:
            chosen = (max_df, max_product, estimated)
    else:
        for max_df, max_product in CAP_TIERS:
            estimated = int(con.execute(f"""
                SELECT coalesce(sum(s.df*t.df),0)
                FROM {s1df_table} s JOIN {tdf_table} t USING(token)
                WHERE s.df <= {max_df} AND t.df <= {max_df}
                  AND s.df*t.df <= {max_product}
            """).fetchone()[0])
            fits = estimated <= MAX_ESTIMATED_PAIRS_PER_FIELD and estimated <= remaining_budget
            print(f'{split} {token_column}: caps df={max_df:,}, product={max_product:,}; '
                  f'estimate={estimated:,}; fits_remaining_budget={fits}')
            if fits and estimated > 0:
                chosen = (max_df, max_product, estimated)
    if chosen is None:
        print(f'SKIP {split} {token_column}: even the smallest useful tier exceeds the safety budget.')
        return None, 0, fixed_caps

    max_df, max_product, estimated = chosen
    con.execute(f"""
        CREATE TEMP TABLE {allowed_table} AS
        SELECT t.candidate_source, t.token
        FROM {s1df_table} s JOIN {tdf_table} t USING(token)
        WHERE s.df <= {max_df} AND t.df <= {max_df}
          AND s.df*t.df <= {max_product}
    """)

    local_path = LOCAL / f'{split}_{token_column}.parquet'
    if local_path.exists():
        local_path.unlink()
    started = time.time()
    con.execute(f"""
        COPY (
            SELECT DISTINCT s.source1_entity_id, t.candidate_entity_id,
                   t.candidate_source, '{token_column}' AS blocking_rule
            FROM {s1_table} s
            JOIN {allowed_table} a USING(token)
            JOIN {target_table} t USING(token, candidate_source)
        ) TO {sql_path(local_path)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    if not local_path.is_file() or not local_path.stat().st_size:
        raise RuntimeError(f'{split} {token_column}: candidate output is missing or empty.')
    drive_path = EXPERIMENT / local_path.name
    shutil.copy2(local_path, drive_path)
    actual_rows = int(con.execute(
        f'SELECT count(*) FROM read_parquet({sql_path(drive_path)})'
    ).fetchone()[0])
    print(f'Saved {drive_path}; rows={actual_rows:,}; estimate={estimated:,}; '
          f'elapsed={time.time()-started:.1f}s')
    return drive_path, estimated, (max_df, max_product)


# -------------------- Progressive recovery and recall audit ------------------
FIELDS_TO_ADD = [
    # Numeric address keys are often both selective and tolerant of name noise.
    'business_address_numbers',
    'business_address_tokens',
    'business_name_numbers',
]
ALL_PATHS = {s: existing_candidate_paths(s) for s in ['train', 'validation']}
REPORT = {
    'target_validation_recall': TARGET_RECALL,
    'cap_tiers': CAP_TIERS,
    'estimated_pair_budgets': {
        'per_field': MAX_ESTIMATED_PAIRS_PER_FIELD,
        'per_split': MAX_ESTIMATED_PAIRS_PER_SPLIT,
    },
    'stages': [],
}
REPORT_PATH = EXPERIMENT / 'recall92_report.json'


def save_report():
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=str))

for split in ['train', 'validation']:
    if not ALL_PATHS[split]:
        raise RuntimeError(f'No nonempty baseline or saved name-token artifact for {split}.')
    initial = recall_metrics(split, ALL_PATHS[split])
    print('Starting coverage:', json.dumps(initial))
    REPORT['stages'].append({'stage': 'baseline_plus_saved_name_tokens', **initial})
    save_report()

for token_column in FIELDS_TO_ADD:
    print(f'\n=== Adding {token_column} ===')
    train_initial = next(x for x in REPORT['stages']
                         if x.get('split') == 'train' and x.get('stage') == 'baseline_plus_saved_name_tokens')
    used_estimate = int(train_initial['candidate_rows_upper_bound']) + sum(
        int(item.get('estimated_pairs', 0)) for item in REPORT['stages']
        if item.get('split') == 'train'
    )
    remaining_train = max(0, MAX_ESTIMATED_PAIRS_PER_SPLIT - used_estimate)
    train_path, train_estimate, selected_caps = build_token_block(
        'train', token_column, remaining_train
    )
    if train_path is None:
        print('No train artifact built; validation pass is skipped to avoid tuning on validation labels.')
        continue

    ALL_PATHS['train'].append((train_path, token_column))
    train_stage = {
        'stage': f'added_{token_column}', 'split': 'train',
        'artifact': str(train_path), 'estimated_pairs': train_estimate,
        'selected_caps_from_train': selected_caps,
        **recall_metrics('train', ALL_PATHS['train']),
    }
    REPORT['stages'].append(train_stage)
    save_report()
    print('Train coverage after field:', json.dumps(train_stage))

    validation_initial = next(x for x in REPORT['stages']
                              if x.get('split') == 'validation' and x.get('stage') == 'baseline_plus_saved_name_tokens')
    used_validation = int(validation_initial['candidate_rows_upper_bound']) + sum(
        int(item.get('estimated_pairs', 0)) for item in REPORT['stages']
        if item.get('split') == 'validation'
    )
    remaining_validation = max(0, MAX_ESTIMATED_PAIRS_PER_SPLIT - used_validation)
    validation_path, validation_estimate, _ = build_token_block(
        'validation', token_column, remaining_validation, fixed_caps=selected_caps
    )
    if validation_path is None:
        print('Validation block skipped because the train-selected caps exceed its configured safety budget.')
        continue
    ALL_PATHS['validation'].append((validation_path, token_column))
    validation_stage = {
        'stage': f'added_{token_column}', 'split': 'validation',
        'artifact': str(validation_path), 'estimated_pairs': validation_estimate,
        'selected_caps_from_train': selected_caps,
        **recall_metrics('validation', ALL_PATHS['validation']),
    }
    REPORT['stages'].append(validation_stage)
    save_report()
    print('Validation coverage after field:', json.dumps(validation_stage))


# -------------------- Disk-backed deduplication for feature input ------------
def candidate_projection(path, tag):
    """Normalize candidate artifact columns and attach this block's provenance."""
    schema = con.execute(f'DESCRIBE SELECT * FROM read_parquet({sql_path(path)})').fetchdf()
    cols = set(schema['column_name'].tolist())
    if not {'source1_entity_id', 'candidate_entity_id'} <= cols:
        raise ValueError(f'Candidate artifact schema not recognized: {path} columns={sorted(cols)}')
    source_expr = (
        'CAST(candidate_source AS VARCHAR)' if 'candidate_source' in cols
        else "CASE WHEN CAST(candidate_entity_id AS VARCHAR) LIKE 'S2-%' THEN 'S2' "
             "WHEN CAST(candidate_entity_id AS VARCHAR) LIKE 'S3-%' THEN 'S3' ELSE NULL END"
    )
    return (
        f"SELECT CAST(source1_entity_id AS VARCHAR) source1_entity_id, "
        f"CAST(candidate_entity_id AS VARCHAR) candidate_entity_id, "
        f"{source_expr} candidate_source, '{tag}' rules_triggered "
        f"FROM read_parquet({sql_path(path)}) "
        "WHERE source1_entity_id IS NOT NULL AND candidate_entity_id IS NOT NULL"
    )


def write_partitioned_union(split, tagged_paths):
    if not tagged_paths:
        raise RuntimeError(f'Cannot write empty {split} candidate union.')
    estimated_gb = sum(p.stat().st_size for p, _tag in tagged_paths) / (1024 ** 3)
    free_gb = shutil.disk_usage('/content').free / (1024 ** 3)
    if free_gb < min(MAX_CANDIDATE_OUTPUT_GB, estimated_gb * 1.5 + 4):
        raise RuntimeError(
            f'Not enough local disk to safely deduplicate {split}: free={free_gb:.1f} GB, '
            f'input artifacts={estimated_gb:.1f} GB. Individual candidate artifacts and recall report remain saved.'
        )

    stage_dir = LOCAL / f'{split}_union_stage'
    raw_dir = stage_dir / 'raw_parts'
    dedup_dir = stage_dir / 'dedup_parts'
    shutil.rmtree(stage_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)
    selects = [candidate_projection(p, tag) for p, tag in tagged_paths]
    source_sql = ' UNION ALL '.join(selects)
    raw_partitioned = raw_dir / 'partitioned'

    print(f'Partitioning {split} union ({estimated_gb:.1f} GB of source artifacts)...')
    con.execute(f"""
        COPY (
            SELECT *, hash(source1_entity_id || chr(31) || candidate_entity_id) % {PARTITIONS} AS bucket
            FROM ({source_sql})
        ) TO {sql_path(raw_partitioned)}
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE)
    """)

    dedup_files = []
    for bucket in range(PARTITIONS):
        bucket_dir = raw_partitioned / f'bucket={bucket}'
        if not bucket_dir.exists():
            continue
        out = dedup_dir / f'part_{bucket:04d}.parquet'
        con.execute(f"""
            COPY (
                SELECT source1_entity_id, candidate_entity_id,
                       any_value(candidate_source) AS candidate_source,
                       string_agg(DISTINCT rules_triggered, ',') AS rules_triggered
                FROM read_parquet({sql_path(str(bucket_dir / '*.parquet'))})
                GROUP BY source1_entity_id, candidate_entity_id
            ) TO {sql_path(out)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        dedup_files.append(out)
        if len(dedup_files) % 16 == 0:
            print(f'{split}: deduplicated {len(dedup_files)} nonempty partitions')

    if not dedup_files:
        raise RuntimeError(f'No candidate rows were written for {split}.')
    final_local = LOCAL / f'final_candidates_{split}.parquet'
    if final_local.exists():
        final_local.unlink()
    dedup_glob = str(dedup_dir / 'part_*.parquet')
    con.execute(f"""
        COPY (SELECT * FROM read_parquet({sql_path(dedup_glob)}))
        TO {sql_path(final_local)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    final_drive = EXPERIMENT / f'final_candidates_{split}.parquet'
    shutil.copy2(final_local, final_drive)
    final_count = int(con.execute(
        f'SELECT count(*) FROM read_parquet({sql_path(final_drive)})'
    ).fetchone()[0])
    REPORT.setdefault('final_artifacts', {})[split] = {
        'path': str(final_drive), 'candidate_pairs': final_count,
    }
    return final_drive


def write_uncovered_positive_diagnostics(split, final_path):
    """Save missed positive pairs with record text so the next blocker is evidence-led."""
    split_value = 'train' if split == 'train' else 'validation'

    def field_expr(source_key, alias, logical, options):
        found = next((name for name in options if name in SOURCE_COLUMNS[source_key]), None)
        if found is None:
            return f'NULL::VARCHAR AS {logical}'
        quoted = '"' + found.replace('"', '""') + '"'
        return f'CAST({alias}.{quoted} AS VARCHAR) AS {logical}'

    s1_name = field_expr('s1', 'a', 'source1_name',
                         ['business_name_norm', 'business_name_raw', 'business_name'])
    s1_address = field_expr('s1', 'a', 'source1_address',
                            ['business_address_norm', 'business_address_raw', 'business_address'])
    s1_country = field_expr('s1', 'a', 'source1_country', ['country_norm', 'country'])
    target_parts = []
    for key, label in [('s2', 'S2'), ('s3', 'S3')]:
        tname = field_expr(key, 'b', 'candidate_name',
                           ['business_name_norm', 'business_name_raw', 'business_name'])
        taddress = field_expr(key, 'b', 'candidate_address',
                              ['business_address_norm', 'business_address_raw', 'business_address'])
        tcountry = field_expr(key, 'b', 'candidate_country', ['country_norm', 'country'])
        target_parts.append(f"""
            SELECT CAST(b.entity_id AS VARCHAR) candidate_entity_id,
                   '{label}' candidate_source, {tname}, {taddress}, {tcountry}
            FROM read_parquet({sql_path(SOURCE_FILES[key])}) b
        """)
    target_sql = ' UNION ALL '.join(target_parts)
    out_local = LOCAL / f'uncovered_{split}_positives.parquet'
    out_drive = EXPERIMENT / f'uncovered_{split}_positives.parquet'
    if out_local.exists():
        out_local.unlink()
    con.execute(f"""
        COPY (
            WITH truth AS (
                SELECT DISTINCT p.source1_entity_id, p.matched_entity_id
                FROM read_csv({sql_path(POSITIVE_FILE)}, delim='\\t', header=true, all_varchar=true) p
                JOIN read_csv({sql_path(SPLIT_FILE)}, delim='\\t', header=true, all_varchar=true) s
                  ON p.source1_entity_id=s.source1_entity_id
                WHERE s.split='{split_value}'
            ), targets AS ({target_sql})
            SELECT t.source1_entity_id, t.matched_entity_id AS candidate_entity_id,
                   x.candidate_source, {s1_name}, {s1_address}, {s1_country},
                   x.candidate_name, x.candidate_address, x.candidate_country
            FROM truth t
            LEFT JOIN read_parquet({sql_path(SOURCE_FILES['s1'])}) a
              ON CAST(a.entity_id AS VARCHAR)=t.source1_entity_id
            LEFT JOIN targets x ON x.candidate_entity_id=t.matched_entity_id
            ANTI JOIN read_parquet({sql_path(final_path)}) c
              ON c.source1_entity_id=t.source1_entity_id
             AND c.candidate_entity_id=t.matched_entity_id
        ) TO {sql_path(out_local)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    shutil.copy2(out_local, out_drive)
    missed = int(con.execute(
        f'SELECT count(*) FROM read_parquet({sql_path(out_drive)})'
    ).fetchone()[0])
    print(f'Saved {missed:,} missed {split} positive pairs with source text:', out_drive)
    return out_drive, missed


for split in ['train', 'validation']:
    print(f'\n=== Final measured {split} recall before union ===')
    final_metrics = recall_metrics(split, ALL_PATHS[split])
    REPORT.setdefault('final_metrics', {})[split] = final_metrics
    save_report()
    print(json.dumps(final_metrics, indent=2))

    final_path = write_partitioned_union(split, ALL_PATHS[split])
    final_metrics_after_union = recall_metrics(split, [(final_path, 'final_union')])
    REPORT.setdefault('final_metrics_after_union', {})[split] = final_metrics_after_union
    print('Recall from written union:', json.dumps(final_metrics_after_union))
    diagnostic_path, missed_count = write_uncovered_positive_diagnostics(split, final_path)
    REPORT.setdefault('uncovered_positive_diagnostics', {})[split] = {
        'path': str(diagnostic_path), 'missed_positive_pairs': missed_count,
    }
    save_report()

validation_recall = REPORT['final_metrics_after_union']['validation']['blocking_recall']
REPORT['target_met'] = bool(validation_recall is not None and validation_recall >= TARGET_RECALL)
REPORT['interpretation'] = (
    f"Measured validation recall reached {validation_recall:.4%}; target {TARGET_RECALL:.0%} met."
    if REPORT['target_met'] else
    f"Measured validation recall was {validation_recall:.4%}; target {TARGET_RECALL:.0%} NOT met. "
    "Do not describe this candidate set as 92% recall. Review per-stage results and add a new blocking family based on observed missed-positive patterns."
)
save_report()
print('\n' + REPORT['interpretation'])
print('Report:', REPORT_PATH)
print('Candidate artifacts:', EXPERIMENT)
print('Feature-engineering inputs: final_candidates_train.parquet and final_candidates_validation.parquet')

if not REPORT['target_met']:
    raise RuntimeError(REPORT['interpretation'])
