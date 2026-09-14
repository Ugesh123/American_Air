"""
Step 2 pipeline: raw twcs.csv -> cleaned, thread-level, time-split, leakage-checked
customer -> AmericanAir resolution pairs.

Usage:
    python3 src/preprocessing/build_dataset.py \
        --input /path/to/twcs.csv \
        --output_dir data/processed

This script is deliberately a single, deterministic, streaming pipeline (no
random seeds needed -- the split is chronological, not random) so it is
reproducible from the raw Kaggle CSV alone.

Memory strategy: the raw file is ~500MB / ~2.8M rows, and the target machine
may have as little as ~4GB RAM. We never load the full file into memory or
into pandas. We do 3 streaming passes over the CSV:
  Pass 1: collect all AmericanAir-authored rows (small: ~37k rows) + the set
          of tweet_ids they reply to.
  Pass 2: collect the full rows for those parent tweet_ids (the customer
          messages), and build a lightweight (tweet_id -> parent_tweet_id)
          index over the WHOLE file (only 2 ints/row) to support thread-root
          backward traversal.
  All further processing operates on the small (~36k-row) AmericanAir-relevant
  subset already extracted, not the full file.
"""
import argparse
import csv
import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.clean import (
    clean_customer_text, clean_agent_text, normalize_for_dedup, is_effectively_empty
)

BRAND = 'AmericanAir'
TS_FMT = '%a %b %d %H:%M:%S %z %Y'
TRAIN_FRAC, DEV_FRAC = 0.70, 0.15  # remaining 0.15 -> eval


def parse_ts(s):
    return datetime.strptime(s, TS_FMT)


def pass1_extract_brand_rows(path):
    aa_rows = {}
    parent_ids_needed = set()
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['author_id'] == BRAND:
                tid = row['tweet_id']
                aa_rows[tid] = row
                parent = row['in_response_to_tweet_id']
                if parent:
                    parent_ids_needed.add(parent)
    return aa_rows, parent_ids_needed


def pass2_extract_parents_and_index(path, parent_ids_needed):
    parent_rows = {}
    parent_index = {}  # tweet_id (int) -> in_response_to_tweet_id (int, -1 if none)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            tid_str = row['tweet_id']
            if tid_str in parent_ids_needed:
                parent_rows[tid_str] = row
            parent_raw = row['in_response_to_tweet_id']
            parent_index[int(tid_str)] = int(parent_raw) if parent_raw else -1
    return parent_rows, parent_index


def build_resolution_pairs(aa_rows, parent_rows):
    pairs = []
    for tid, aa_row in aa_rows.items():
        parent_id = aa_row['in_response_to_tweet_id']
        if not parent_id:
            continue
        parent_row = parent_rows.get(parent_id)
        if parent_row is None or parent_row['inbound'] != 'True':
            continue
        pairs.append((parent_row, aa_row))
    return pairs


def find_root(tweet_id, parent_index, max_depth=100):
    seen = set()
    cur = tweet_id
    depth = 0
    while depth < max_depth:
        if cur in seen:
            return cur
        seen.add(cur)
        parent = parent_index.get(cur, -1)
        if parent == -1:
            return cur
        cur = parent
        depth += 1
    return cur


def run(input_path, output_dir):
    stats = {}

    print('[Pass 1] Scanning for AmericanAir rows...')
    aa_rows, parent_ids_needed = pass1_extract_brand_rows(input_path)
    stats['aa_outbound_rows'] = len(aa_rows)
    print(f'  AmericanAir outbound rows: {len(aa_rows)}')

    print('[Pass 2] Fetching parent rows + building full parent-link index...')
    parent_rows, parent_index = pass2_extract_parents_and_index(input_path, parent_ids_needed)
    stats['parent_rows_found'] = len(parent_rows)
    stats['parent_ids_requested'] = len(parent_ids_needed)
    print(f'  Parent rows found: {len(parent_rows)} / {len(parent_ids_needed)} requested')
    print(f'  Full-file parent index size: {len(parent_index)} tweet_ids')

    raw_pairs = build_resolution_pairs(aa_rows, parent_rows)
    stats['raw_resolution_pairs'] = len(raw_pairs)
    print(f'[Pairs] Raw customer -> AmericanAir resolution pairs: {len(raw_pairs)}')

    # thread roots
    pair_records = []
    for c, a in raw_pairs:
        root = find_root(int(c['tweet_id']), parent_index)
        pair_records.append({
            'customer_tweet_id': c['tweet_id'], 'customer_author_id': c['author_id'],
            'customer_created_at': c['created_at'], 'customer_text': c['text'],
            'customer_in_response_to': c['in_response_to_tweet_id'],
            'aa_tweet_id': a['tweet_id'], 'aa_created_at': a['created_at'], 'aa_text': a['text'],
            'thread_root_id': root,
        })
    n_threads_raw = len(set(r['thread_root_id'] for r in pair_records))
    stats['raw_thread_count'] = n_threads_raw
    print(f'[Threads] Distinct thread roots (pre-clean): {n_threads_raw}')

    # ---- cleaning ----
    cleaned = []
    n_empty_customer = n_empty_agent = 0
    for r in pair_records:
        cust_clean = clean_customer_text(r['customer_text'])
        agent_clean = clean_agent_text(r['aa_text'])
        if is_effectively_empty(cust_clean):
            n_empty_customer += 1
            continue
        if is_effectively_empty(agent_clean):
            n_empty_agent += 1
            continue
        cleaned.append({
            **r,
            'customer_text_clean': cust_clean,
            'aa_text_clean': agent_clean,
            'customer_created_at_parsed': parse_ts(r['customer_created_at']).isoformat(),
        })
    stats['removed_empty_customer'] = n_empty_customer
    stats['removed_empty_agent'] = n_empty_agent
    print(f'[Clean] Removed {n_empty_customer} empty-customer-text pairs, '
          f'{n_empty_agent} empty-agent-text pairs')

    # ---- exact-dup removal ----
    seen_exact, deduped = set(), []
    n_exact_dupes = 0
    for r in cleaned:
        key = (r['customer_text_clean'], r['aa_text_clean'])
        if key in seen_exact:
            n_exact_dupes += 1
            continue
        seen_exact.add(key)
        deduped.append(r)
    stats['removed_exact_duplicates'] = n_exact_dupes
    stats['pairs_after_cleaning'] = len(deduped)
    print(f'[Clean] Removed {n_exact_dupes} exact-duplicate pairs')
    print(f'[Clean] Pairs remaining after full cleaning: {len(deduped)}')

    # ---- thread-level time-based split ----
    threads = defaultdict(list)
    for p in deduped:
        threads[p['thread_root_id']].append(p)
    thread_min_ts = {root: min(datetime.fromisoformat(m['customer_created_at_parsed']) for m in members)
                      for root, members in threads.items()}
    sorted_roots = sorted(thread_min_ts, key=lambda r: thread_min_ts[r])

    total_pairs = len(deduped)
    train_target = total_pairs * TRAIN_FRAC
    dev_target = total_pairs * (TRAIN_FRAC + DEV_FRAC)
    split_assignment, running = {}, 0
    for root in sorted_roots:
        n = len(threads[root])
        if running < train_target:
            split_assignment[root] = 'train'
        elif running < dev_target:
            split_assignment[root] = 'dev'
        else:
            split_assignment[root] = 'eval'
        running += n

    for p in deduped:
        p['split'] = split_assignment[p['thread_root_id']]

    split_pairs = defaultdict(list)
    for p in deduped:
        split_pairs[p['split']].append(p)

    stats['splits'] = {}
    for split in ['train', 'dev', 'eval']:
        sp = split_pairs[split]
        n_threads = len(set(p['thread_root_id'] for p in sp))
        ts_list = [datetime.fromisoformat(p['customer_created_at_parsed']) for p in sp]
        stats['splits'][split] = {
            'pairs': len(sp), 'threads': n_threads,
            'date_min': min(ts_list).isoformat(), 'date_max': max(ts_list).isoformat(),
        }
        print(f'[Split] {split}: {len(sp)} pairs, {n_threads} threads, '
              f'{min(ts_list).date()} -> {max(ts_list).date()}')

    # ---- leakage checks ----
    thread_to_splits = defaultdict(set)
    for p in deduped:
        thread_to_splits[p['thread_root_id']].add(p['split'])
    thread_violations = sum(1 for s in thread_to_splits.values() if len(s) > 1)
    stats['leakage_thread_violations'] = thread_violations
    print(f'[Leakage] Threads spanning >1 split: {thread_violations} (must be 0)')
    assert thread_violations == 0, 'LEAKAGE DETECTED: a thread spans multiple splits'

    train_cust_norm = set(normalize_for_dedup(p['customer_text_clean']) for p in split_pairs['train'])
    train_aa_norm = set(normalize_for_dedup(p['aa_text_clean']) for p in split_pairs['train'])

    stats['leakage_cross_split_text'] = {}
    for split in ['dev', 'eval']:
        cust_flags = sum(1 for p in split_pairs[split]
                          if normalize_for_dedup(p['customer_text_clean']) in train_cust_norm)
        aa_flags = sum(1 for p in split_pairs[split]
                        if normalize_for_dedup(p['aa_text_clean']) in train_aa_norm)
        stats['leakage_cross_split_text'][split] = {
            'dup_customer_text_in_train': cust_flags, 'dup_aa_text_in_train': aa_flags,
            'total': len(split_pairs[split]),
        }
        print(f'[Leakage] {split}: {cust_flags}/{len(split_pairs[split])} customer texts '
              f'and {aa_flags}/{len(split_pairs[split])} AA-reply texts duplicate normalized '
              f'strings seen in train (flagged, not removed -- see README for rationale)')

    # ---- write outputs ----
    os.makedirs(output_dir, exist_ok=True)
    thread_size = {root: len(members) for root, members in threads.items()}

    def pair_id(p):
        return f"aa_{p['customer_tweet_id']}_{p['aa_tweet_id']}"

    fname = {'train': 'train.jsonl', 'dev': 'dev.jsonl', 'eval': 'eval_pool.jsonl'}
    for split in ['train', 'dev', 'eval']:
        out_path = os.path.join(output_dir, fname[split])
        with open(out_path, 'w', encoding='utf-8') as f:
            for p in split_pairs[split]:
                cust_norm = normalize_for_dedup(p['customer_text_clean'])
                aa_norm = normalize_for_dedup(p['aa_text_clean'])
                record = {
                    'pair_id': pair_id(p), 'thread_id': p['thread_root_id'], 'split': p['split'],
                    'thread_size_pairs': thread_size[p['thread_root_id']],
                    'customer_tweet_id': p['customer_tweet_id'],
                    'customer_author_id': p['customer_author_id'],
                    'customer_created_at_raw': p['customer_created_at'],
                    'customer_text_raw': p['customer_text'],
                    'customer_text_clean': p['customer_text_clean'],
                    'aa_tweet_id': p['aa_tweet_id'],
                    'aa_created_at_raw': p['aa_created_at'],
                    'aa_text_raw': p['aa_text'],
                    'aa_text_clean': p['aa_text_clean'],
                    'is_first_turn': not bool(p['customer_in_response_to']),
                    'dup_customer_text_in_train': (split != 'train') and (cust_norm in train_cust_norm),
                    'dup_aa_text_in_train': (split != 'train') and (aa_norm in train_aa_norm),
                }
                f.write(json.dumps(record) + '\n')
        print(f'[Write] {out_path}: {len(split_pairs[split])} records')

    with open(os.path.join(output_dir, '..', 'step2_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f'[Write] step2_stats.json')

    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='path to raw twcs.csv')
    ap.add_argument('--output_dir', default='data/processed')
    args = ap.parse_args()
    run(args.input, args.output_dir)
