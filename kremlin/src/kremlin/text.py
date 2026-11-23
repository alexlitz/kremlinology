"""Built-in prompts and fresh filler text (no datasets to download).

prompts(n, seed): varied short prompts whose first token is genuinely uncertain, for
  fingerprints and output comparisons. Use --prompts FILE for a real dataset average.
prose(chars, generator): fresh, readable pseudo-prose with a random nonce first, for cache
  probes. Readable sentences avoid the content filters random word salad trips.
"""
import json
import random
import secrets
from pathlib import Path

TEMPLATES = [
    'Name a {category}. Reply with one word only.',
    'Write the first sentence of a short story about {topic}.',
    'Continue this sentence with a few words: "{opening}"',
    'Explain {concept} in one sentence.',
    'Suggest a name for {thing}. Reply with the name only.',
    'Give one surprising fact about {topic}.',
    'Write a one-line poem about {topic}.',
    'Describe {topic} using three adjectives.',
]
FILL = dict(
    category=['color', 'animal', 'fruit', 'city', 'musical instrument', 'tree', 'gemstone', 'board game', 'spice',
              'constellation', 'river', 'vegetable', 'flower', 'metal', 'dance', 'bird'],
    topic=['a lighthouse keeper', 'an abandoned train station', 'a desert at night', 'a clockmaker', 'the deep sea',
           'a mountain village', 'a lost map', 'an old library', 'a thunderstorm', 'a city garden', 'a glacier',
           'a night market', 'a migrating bird', 'a broken radio', 'a bakery at dawn', 'a winter harbor'],
    opening=['The last thing she remembered was', 'Every morning the river', 'Nobody in the village knew why',
             'The machine hummed softly as', 'At the edge of the map there was', 'He opened the letter and',
             'When the lights went out,', 'The garden had been empty since'],
    concept=['entropy', 'photosynthesis', 'compound interest', 'plate tectonics', 'recursion', 'inflation',
             'natural selection', 'a hash function', 'supply and demand', 'the placebo effect'],
    thing=['a coffee shop', 'a pet turtle', 'a sailing boat', 'a new planet', 'a jazz band', 'a robot', 'a bakery'],
)

WORDS = dict(
    subject=['The archive', 'A small committee', 'The northern station', 'Every surveyor', 'The old harbor',
             'A visiting engineer', 'The regional council', 'Most travellers', 'The river authority', 'Our guide'],
    verb=['recorded', 'described', 'reviewed', 'repaired', 'measured', 'catalogued', 'questioned', 'restored',
          'mapped', 'compared'],
    object=['the eastern bridge', 'a collection of letters', 'the winter supply routes', 'several damaged maps',
            'the tide tables', 'an unusual inscription', 'the market accounts', 'the lighthouse lamps',
            'a list of local names', 'the forest boundaries'],
    clause=['before the spring floods arrived', 'during a long and quiet season', 'with help from the villagers',
            'after the second survey was finished', 'while the roads were still closed', 'for the first time in years',
            'despite the difficult weather', 'as the old records suggested'],
)


def prompts(n=60, seed=0, path=None):
    if path:
        text = Path(path).read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()] if path.endswith('.jsonl') else \
            [line.strip() for line in text.splitlines() if line.strip()]
        items = [r['text'] if isinstance(r, dict) else r for r in rows]
        return items[:n] if n else items
    generator = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        template = generator.choice(TEMPLATES)
        filled = template.format(**{k: generator.choice(v) for k, v in FILL.items()})
        if filled not in seen:
            seen.add(filled)
            out.append(filled)
        if len(seen) > 2000:
            break
    return out


def fresh_salt(salt=None):
    """Cache probes take a salt: fresh by default so documents never repeat across runs (a document an
    earlier run cached would count as a false hit); the CLI records it so an audit regenerates the
    exact same requests."""
    return salt or secrets.token_hex(6)


def prose(chars, generator, salt):
    parts = [f'Reference document {salt}-{generator.getrandbits(40):010x}.']
    total = len(parts[0])
    while total < chars:
        sentence = (f"{generator.choice(WORDS['subject'])} {generator.choice(WORDS['verb'])} "
                    f"{generator.choice(WORDS['object'])} {generator.choice(WORDS['clause'])}.")
        parts.append(sentence)
        total += len(sentence) + 1
    return ' '.join(parts)
