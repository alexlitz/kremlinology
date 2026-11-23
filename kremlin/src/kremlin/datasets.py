"""Build prompt files from public datasets (same sampling as the research campaigns). Needs `pip install datasets`.

  codealpaca   CodeAlpaca-20k instructions (+ input), 20-600 characters
  wikitext     wikitext-103 validation paragraphs cut to a 30-60 word continuation prompt
  hf:REPO:SPLIT:FIELD   any Hugging Face dataset text field
Prompts are drawn uniformly without replacement with a fixed seed; each line records its source index so a
dataset average can be reproduced and audited.
"""
import json
import random


def build(source, n, seed=0):
    from datasets import load_dataset
    generator = random.Random(seed)
    out = []
    if source == 'codealpaca':
        data = load_dataset('sahil2801/CodeAlpaca-20k', split='train')
        order = list(range(len(data)))
        generator.shuffle(order)
        for index in order:
            row = data[index]
            body = row['instruction'].strip() + (('\n\n' + row['input'].strip()) if row['input'].strip() else '')
            if 20 <= len(body) <= 600:
                out.append(dict(text=body, source='sahil2801/CodeAlpaca-20k:train', index=index))
            if len(out) == n:
                break
    elif source == 'wikitext':
        data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
        paragraphs = [i for i, row in enumerate(data) if len(row['text']) > 300 and not row['text'].strip().startswith('=')]
        generator.shuffle(paragraphs)
        for index in paragraphs[:n]:
            words = data[index]['text'].split()
            snippet = ' '.join(words[:generator.randint(30, 60)])
            out.append(dict(text='Continue the following passage. Reply with the continuation only.\n\n' + snippet,
                            source='wikitext-103-raw-v1:validation', index=index))
    elif source.startswith('hf:'):
        _, repo, split, field = source.split(':', 3)
        data = load_dataset(repo, split=split)
        order = list(range(len(data)))
        generator.shuffle(order)
        for index in order[:n]:
            value = data[index][field]
            if isinstance(value, str) and value.strip():
                out.append(dict(text=value.strip(), source=f'{repo}:{split}:{field}', index=index))
    else:
        raise ValueError('source must be codealpaca, wikitext or hf:REPO:SPLIT:FIELD')
    return out


def write(items, path):
    with open(path, 'w') as handle:
        for item in items:
            handle.write(json.dumps(item) + '\n')
