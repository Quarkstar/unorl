#!/usr/bin/env python3
"""Prepare and launch the sampled benchmark comparison, without training."""
import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from train import ROOT, CODE, PYTHON, environment, PIN


def prepare():
    import pandas as pd
    model = ROOT / 'models/Qwen2.5-Math-1.5B-system'
    model.mkdir(exist_ok=True)
    for source in (ROOT / 'models/Qwen2.5-Math-1.5B').iterdir():
        target = model / source.name
        if source.is_file() and source.name != 'config.json' and not target.exists():
            target.symlink_to(source.resolve())
    cfg = json.loads((ROOT / 'models/Qwen2.5-Math-1.5B/config.json').read_text())
    cfg['max_position_embeddings'] = 16384
    (model / 'config.json').write_text(json.dumps(cfg, indent=2))
    raw = ROOT / 'data/raw/benchmark-audit'
    math = [json.loads(x) for x in (raw / 'HuggingFaceH4--MATH-500--test.jsonl').read_text().splitlines()]
    amc = pd.read_parquet(raw / 'math-ai--amc23--test-00000-of-00001.parquet').to_dict('records')
    aime = [json.loads(x) for x in (ROOT / 'data/raw/aime25/test.jsonl').read_text().splitlines()]
    for name, records in [('math500', math), ('amc23', amc), ('aime25', aime)]:
        rows = []
        for i, record in enumerate(records):
            problem = record.get('problem', record.get('question'))
            content = ('Solve the following math problem step by step. The last line of your response '
                       'should be of the form Answer: $Answer (without quotes) where $Answer is the '
                       'answer to the problem.\n\n' + problem + '\n\nRemember to put your answer '
                       'on its own line after "Answer:".')
            rows.append(dict(prompt=[dict(role='user', content=content)], data_source=name,
                             reward_model=dict(ground_truth=str(record['answer']), style='rule'),
                             extra_info=dict(index=str(i))))
        pd.DataFrame(rows).to_parquet(ROOT / f'data/{name}-benchmark.parquet', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    env = environment()
    env['PYTHONPATH'] = str(ROOT / '.deps') + ':' + env['PYTHONPATH']
    if not args.worker:
        run_id = 'system-benchmarks-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        run = ROOT / 'runs' / run_id
        run.mkdir()
        for folder in ['conditional_rl', 'scripts', 'configs']:
            shutil.copytree(CODE / folder, run / 'source' / folder,
                            ignore=shutil.ignore_patterns('__pycache__'))
        log = ROOT / 'runs/logs' / (run_id + '.log')
        with log.open('x') as handle:
            process = subprocess.Popen([str(PYTHON), str(run / 'source/scripts/evaluate.py'),
                                        '--worker', '--run-id', run_id],
                                       env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        (run / 'pid').write_text(str(process.pid))
        print(json.dumps(dict(run_id=run_id, pid=process.pid, log=str(log))))
        return
    run = ROOT / 'runs' / args.run_id
    try:
        prepare()
        cfg = json.loads((CODE / 'configs/experiment.json').read_text())
        cfg.update({'trainer.policy.model.path': str(ROOT / 'models/Qwen2.5-Math-1.5B-system'),
                    'trainer.placement.colocate_all': False,
                    'data.val_data': [str(ROOT / 'data/math500-benchmark.parquet')],
                    'trainer.export_path': str(run / 'exports'),
                    'trainer.log_path': str(run / 'infra'), 'trainer.run_name': args.run_id,
                    'trainer.eval_interval': 1, 'trainer.eval_batch_size': 128,
                    'trainer.max_prompt_length': 2048,
                    'trainer.dump_eval_results': True,
                    'environment.env_class': 'benchmark_math',
                    'environment.skyrl_gym.max_env_workers': 0})
        (run / 'config.json').write_text(json.dumps(cfg, indent=2))
        (run / 'skyrl-commit.txt').write_text(PIN)
        for audit in (ROOT / 'data/raw/benchmark-audit').glob('*.audit.json'):
            shutil.copy2(audit, run / audit.name)
        command = [str(PYTHON), '-m', 'conditional_rl.benchmark_eval'] + [
            f'{key}={json.dumps(value)}' for key, value in cfg.items()]
        status = subprocess.call(command, cwd=ROOT, env=env)
    except Exception:
        (run / 'exit-status').write_text('1\n')
        raise
    (run / 'exit-status').write_text(str(status) + '\n')
    raise SystemExit(status)


if __name__ == '__main__':
    main()
