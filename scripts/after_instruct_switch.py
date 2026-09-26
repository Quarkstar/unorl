"""After instruct GRPO finishes, remove weights, preserve results, launch base pair."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PAIR = ROOT / 'runs/qwen3-4b-instruct-pair-20260915-02'
PYTHON = ROOT.parent / 'SciBuddy/.venv-skyrl/bin/python'
STATUS = ROOT / 'runs/logs/qwen3-4b-instruct-grpo-20260915-02.exit-status'
BASE = ROOT / 'models/Qwen3-4B-Base'
print('Waiting for instruct GRPO to finish successfully (conditional was dequeued).', flush=True)
while not STATUS.exists():
    time.sleep(30)
if STATUS.read_text().strip() != '0':
    raise RuntimeError('Current Qwen3 instruct GRPO did not finish successfully')
print('GRPO completed. Waiting for verified Qwen/Qwen3-4B-Base download.', flush=True)
while not (BASE / 'download-audit.json').exists():
    time.sleep(30)
audit = json.loads((BASE / 'download-audit.json').read_text())
assert audit['repo_id'] == 'Qwen/Qwen3-4B-Base', audit['repo_id']
assert audit.get('files'), 'Missing checksum verification'
from train import environment
env = environment()
validation = '''
from transformers import AutoTokenizer, AutoConfig
from conditional_rl.system_conditioning import system_tokenizer, prefix_ids, relabel_output
p='models/Qwen3-4B-Base'
assert AutoConfig.from_pretrained(p,local_files_only=True).model_type=='qwen3'
t=AutoTokenizer.from_pretrained(p,local_files_only=True)
for label in [None,'CORRECT']:
 s=system_tokenizer(t,label)
 ids=s.apply_chat_template([{'role':'user','content':'What is 2+2?'}],tokenize=True,add_generation_prompt=True,return_dict=False)
 if label:
  assert ids[:len(prefix_ids(t,label))]==prefix_ids(t,label)
  out,labels=relabel_output({'prompt_token_ids':[ids,ids],'rewards':[1,-1]},t)
  assert labels==['CORRECT','INCORRECT']
print('Base model tokenizer and conditioning checks passed')
'''
subprocess.run([str(PYTHON), '-c', validation], cwd=ROOT, env=env, check=True)
removed = []
weight_suffixes = {'.pt', '.pth', '.bin', '.safetensors', '.distcp', '.ckpt'}
for run in ROOT.joinpath('runs').iterdir():
    if not run.is_dir() or not run.name.startswith('qwen3-4b-instruct-'):
        continue
    # Select only weight/optimizer payloads in known checkpoint/export locations.
    # Logs, metrics, plots, configs, source snapshots and evaluation dumps remain.
    for folder in [run / 'checkpoints', run / 'exports']:
        if not folder.exists():
            continue
        for p in folder.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix in weight_suffixes:
                removed.append({'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size})
                p.unlink()
(ROOT / 'runs/qwen3-base-cleanup.json').write_text(json.dumps(
    {'reason': 'User requested removal of checkpoint and weight payloads only',
     'preserved': 'logs, metrics, curves, configs, source snapshots, evaluation results',
     'removed': removed}, indent=2) + '\n')
command = [str(PYTHON), 'scripts/train.py', '--launch', '--pair', '--baseline-first',
           '--config', 'qwen3-4b-base.json', '--run-id', 'qwen3-4b-base-pair-20260916-01']
result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
print(result.stdout, flush=True)
