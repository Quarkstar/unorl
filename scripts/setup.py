#!/usr/bin/env python3
"""Download pinned inputs locally and create a tokenizer/model with two control tokens."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'


def main():
    import torch
    import pandas as pd
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest_path = ROOT / 'assets.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sources = [
        ('model', 'Qwen/Qwen2.5-Math-1.5B', 'model', 'models/Qwen2.5-Math-1.5B'),
        ('train_original', 'BytedTsinghua-SIA/DAPO-Math-17k', 'dataset', 'data/raw/dapo'),
        ('train', 'eshwarprasadS/DAPO-Math-8k-Stratified', 'dataset', 'data/raw/dapo-stratified'),
        ('eval', 'BytedTsinghua-SIA/AIME-2024', 'dataset', 'data/raw/aime'),
        ('eval25', 'math-ai/aime25', 'dataset', 'data/raw/aime25'),
        ('eval26', 'math-ai/aime26', 'dataset', 'data/raw/aime26'),
    ]
    for key, repo, kind, relative in sources:
        previous = manifest.get(key, {})
        revision = (previous.get('revision') if previous.get('repo') == repo else None) or HfApi().repo_info(repo, repo_type=kind).sha
        manifest[key] = dict(repo=repo, revision=revision, path=relative)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Downloading {repo}@{revision}', flush=True)
        snapshot_download(repo, repo_type=kind, revision=revision, local_dir=ROOT / relative,
                          allow_patterns=['*.jsonl', '*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', '*.parquet', '*.md'],
                          max_workers=4)

    # Match SkyRL's DAPO cleaning: deduplicate identical rows and discard conflicting labels.
    for key, filename in [('eval', 'aime-2024')]:
        frame = pd.read_parquet(ROOT / manifest[key]['path'] / 'data' / f'{filename}.parquet')
        def canonical(value):
            if hasattr(value, 'tolist'):
                value = value.tolist()
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=lambda x: x.tolist())
        frame['_prompt'] = frame.prompt.map(canonical)
        frame['_reward'] = frame.reward_model.map(canonical)
        frame = frame.drop_duplicates(['data_source', '_prompt', 'ability', '_reward'])
        frame = frame[frame.groupby('_prompt')['_reward'].transform('nunique') == 1]
        frame = frame.drop(columns=['_prompt', '_reward']).reset_index(drop=True)
        frame.to_parquet(ROOT / 'data' / f'{key}.parquet', index=False)
        manifest[key]['cleaned_rows'] = len(frame)

    folder = ROOT / manifest['train']['path']
    for split, count, output_name in [('train', 7500, 'train'), ('validation', 500, 'dapo-validation')]:
        filename = 'train.parquet' if split == 'train' else 'val.parquet'
        files = [folder / 'data' / filename]
        frame = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
        if len(frame) != count:
            raise ValueError(f'Expected published {split} split with {count} rows, found {len(frame)}')
        frame.to_parquet(ROOT / 'data' / f'{output_name}.parquet', index=False)
        manifest['train'][f'{split}_rows'] = len(frame)
    manifest['train']['split'] = 'train'
    manifest['train']['prepared_path'] = 'data/train.parquet'

    evaluation = pd.read_parquet(ROOT / 'data/eval.parquet')
    evaluation['data_source'] = 'aime2024'
    evaluations = [evaluation]
    header = 'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.'
    suffix = 'Remember to put your answer on its own line after "Answer:".'
    for key, year in [('eval25', 2025), ('eval26', 2026)]:
        folder = ROOT / manifest[key]['path']
        files = sorted(folder.glob('*.jsonl')) + sorted(folder.glob('*.parquet')) + sorted(folder.glob('data/*.parquet'))
        raw = pd.concat([pd.read_json(file, lines=True) if file.suffix == '.jsonl' else pd.read_parquet(file)
                         for file in files], ignore_index=True)
        if len(raw) != 30:
            raise ValueError(f'Expected 30 AIME {year} questions, found {len(raw)}')
        rows = []
        for index, row in raw.iterrows():
            problem = str(row['problem'])
            answer = str(row['answer'])
            rows.append(dict(data_source=f'aime{year}',
                             prompt=[dict(role='user', content=header + '\n\n' + problem + '\n\n' + suffix)],
                             ability='MATH', reward_model=dict(ground_truth=answer, style='rule'),
                             extra_info=dict(index=str(index))))
        evaluations.append(pd.DataFrame(rows))
        manifest[key]['cleaned_rows'] = len(rows)
    combined = pd.concat(evaluations, ignore_index=True)
    combined['extra_info'] = [dict(index=str(index)) for index in range(len(combined))]
    combined.to_parquet(ROOT / 'data/eval.parquet', index=False)
    def prompt_key(prompt):
        return ''.join(''.join(message['content'].split()) for message in prompt)
    audit_frames = {name: pd.read_parquet(ROOT / 'data' / f'{name}.parquet')
                    for name in ['train', 'dapo-validation', 'eval']}
    keys = {name: set(frame.prompt.map(prompt_key)) for name, frame in audit_frames.items()}
    audit = dict(rows={name: len(frame) for name, frame in audit_frames.items()},
                 unique_whitespace_normalized_prompts={name: len(value) for name, value in keys.items()},
                 train_validation_overlap=len(keys['train'] & keys['dapo-validation']),
                 train_aime_overlap=len(keys['train'] & keys['eval']))
    (ROOT / 'data/audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if audit['train_aime_overlap']:
        raise ValueError('Training prompts overlap AIME evaluation; inspect data/audit.json')
    manifest['evaluation'] = dict(path='data/eval.parquet', rows=len(combined), years=[2024, 2025, 2026])

    base = ROOT / manifest['model']['path']
    output = ROOT / 'models/Qwen2.5-Math-1.5B-conditional'
    if not (output / 'conditioning.json').exists():
        torch.manual_seed(42)
        tokenizer = AutoTokenizer.from_pretrained(base)
        if not tokenizer.chat_template:
            tokenizer.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.add_special_tokens({'additional_special_tokens': ['<|CORRECT|>', '<|INCORRECT|>']})
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        # Qwen reserves unused embedding rows. Resize only when those are exhausted.
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        with torch.no_grad():
            for label in ['CORRECT', 'INCORRECT']:
                token_id = tokenizer.convert_tokens_to_ids(f'<|{label}|>')
                seed_ids = tokenizer.encode(label, add_special_tokens=False)
                for embedding in [model.get_input_embeddings(), model.get_output_embeddings()]:
                    embedding.weight[token_id].copy_(embedding.weight[seed_ids].mean(dim=0))
        tokenizer.save_pretrained(output)
        model.save_pretrained(output)
        info = {label: tokenizer.convert_tokens_to_ids(f'<|{label}|>') for label in ['CORRECT', 'INCORRECT']}
        (output / 'conditioning.json').write_text(json.dumps(info, indent=2) + '\n')
    config_path = output / 'config.json'
    model_config = json.loads(config_path.read_text())
    model_config['max_position_embeddings'] = 16384
    config_path.write_text(json.dumps(model_config, indent=2) + '\n')
    manifest['context_extension'] = dict(original=4096, configured=16384, rope_scaling=None)
    manifest['conditional_model'] = str(output.relative_to(ROOT))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Assets ready:', manifest_path, flush=True)


if __name__ == '__main__':
    main()
