"""Compare the active GRPO run at step 20 and execute the authorized fallback."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
WATCH = ROOT / 'runs/step20-model-followup-20260915'
GRPO = ROOT / 'runs/qwen25-math-1.5b-grpo-20260915-02'
COND = ROOT / 'runs/qwen25-math-1.5b-conditional-20260915-02'
PYTHON = ROOT.parent / 'SciBuddy/.venv-skyrl/bin/python'
RULES = {'eval/math500/accuracy': 0.05, 'eval/amc23/pass@8': 0.10,
         'eval/aime25/pass@8': 2 / 30}


def save(name, value):
    target = WATCH / name
    temporary = target.with_suffix(target.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(target)


def status(stage, **kwargs):
    value = dict(stage=stage, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    save('status.json', value)
    print(json.dumps(value), flush=True)


def step20(path):
    try:
        rows = [json.loads(x) for x in path.read_text().splitlines()]
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return next((r for r in rows if r['step'] == 20 and all(k in r['metrics'] for k in RULES)), None)


def main():
    pid = int((ROOT / 'runs/logs/qwen25-math-1.5b-grpo-20260915-02.pid').read_text())
    parent = psutil.Process(pid)
    assert 'qwen25-math-1.5b-grpo-20260915-02' in ' '.join(parent.cmdline())
    assert os.getpgid(pid) == pid
    status('waiting_for_grpo_step20')
    deadline = time.monotonic() + 6 * 3600
    while (grpo := step20(GRPO / 'metrics.jsonl')) is None:
        if not parent.is_running() or parent.status() == psutil.STATUS_ZOMBIE:
            raise RuntimeError('GRPO exited before step-20 results were saved')
        if time.monotonic() > deadline:
            raise RuntimeError('Timed out waiting for step-20 evaluation')
        time.sleep(0.5)
    cond = step20(COND / 'metrics.jsonl')
    assert cond is not None
    comparison = {k: dict(conditional=cond['metrics'][k], grpo=grpo['metrics'][k],
                         difference=grpo['metrics'][k] - cond['metrics'][k], tolerance=v)
                  for k, v in RULES.items()}
    similar = all(abs(v['difference']) <= v['tolerance'] + 1e-10 for v in comparison.values())
    save('comparison.json', dict(step=20, similar_by_practical_rule=similar,
         note='Screening rule, not statistical equivalence; each condition has its own initial score.',
         benchmarks=comparison, conditional=cond, grpo=grpo))
    if not similar:
        status('not_similar_grpo_continues', comparison=comparison)
        return

    assert (GRPO / 'checkpoints/global_step_20').is_dir()
    status('similar_stopping_qwen25_grpo', comparison=comparison)
    owned = parent.children(recursive=True) + [parent]
    os.killpg(pid, signal.SIGTERM)
    _, alive = psutil.wait_procs(owned, timeout=20)
    for p in alive:
        try:
            if p.status() != psutil.STATUS_ZOMBIE:
                p.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=10)
    assert not [p for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
    (GRPO / 'stop-audit.json').write_text(json.dumps(dict(
        reason='User-authorized Qwen3 fallback after similar step-20 results',
        checkpoint='checkpoints/global_step_20', stopped_pids=[p.pid for p in owned]), indent=2) + '\n')

    status('downloading_qwen3')
    from huggingface_hub import HfApi, snapshot_download
    repo = 'Qwen/Qwen3-4B-Instruct-2507'
    model = ROOT / 'models/Qwen3-4B-Instruct-2507'
    audit_path = model / 'download-audit.json'
    download_pid_file = ROOT / 'runs/logs/qwen3-download.pid'
    if download_pid_file.exists() and not audit_path.exists():
        downloader = psutil.Process(int(download_pid_file.read_text()))
        status('waiting_for_existing_qwen3_download')
        while not audit_path.exists():
            if not downloader.is_running() or downloader.status() == psutil.STATUS_ZOMBIE:
                raise RuntimeError('Existing Qwen3 download ended without a verified model')
            time.sleep(5)
    revision = (json.loads(audit_path.read_text())['revision'] if audit_path.exists()
                else HfApi().model_info(repo).sha)
    snapshot_download(repo_id=repo, revision=revision, local_dir=model,
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', 'README.md', 'LICENSE*'],
                      max_workers=4)
    audit = dict(repo_id=repo, revision=revision, local_dir=str(model))
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
    else:
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
    save('model-assets.json', audit)

    # Validate the new model's actual tokenizer/template before GPU work starts.
    import sys
    sys.path.insert(0, str(WATCH / 'source/scripts'))
    from train import environment
    env = environment()
    validation = '''
from transformers import AutoTokenizer, AutoConfig
from conditional_rl.system_conditioning import system_tokenizer, prefix_ids, relabel_output
from pathlib import Path
p=Path('models/Qwen3-4B-Instruct-2507')
c=AutoConfig.from_pretrained(p, local_files_only=True)
assert c.model_type == 'qwen3'
t=AutoTokenizer.from_pretrained(p, local_files_only=True)
tokens=system_tokenizer(t).apply_chat_template([{'role':'user','content':'What is 2+2?'}], tokenize=True, add_generation_prompt=True, return_dict=False)
assert tokens[:len(prefix_ids(t,'CORRECT'))] == prefix_ids(t,'CORRECT')
o, labels=relabel_output({'prompt_token_ids':[tokens,tokens], 'rewards':[1,-1]},t)
assert labels == ['CORRECT','INCORRECT']
assert o['prompt_token_ids'][1][:len(prefix_ids(t,'INCORRECT'))] == prefix_ids(t,'INCORRECT')
print('Qwen3 config and system-conditioning tokenizer checks passed')
'''
    result = subprocess.run([str(PYTHON), '-c', validation], cwd=ROOT, env=env,
                            text=True, capture_output=True)
    (WATCH / 'model-validation.log').write_text(result.stdout + result.stderr)
    result.check_returncode()
    status('launching_qwen3_pair')
    command = [str(PYTHON), str(WATCH / 'source/scripts/train.py'), '--launch', '--pair',
               '--baseline-first', '--config', 'qwen3-4b-instruct.json',
               '--run-id', 'qwen3-4b-instruct-pair-20260915-01']
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    launch = json.loads(result.stdout)
    save('qwen3-launch.json', launch)
    status('qwen3_pair_launched', launch=launch)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=repr(error))
        raise
