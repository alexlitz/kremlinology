"""Model-independent recipes; explicit messages can replace any template."""

RECIPES = {
    "first_token": {
        "messages": [
            {
                "role": "user",
                "content": "Name one common fruit. Reply with its name only.",
            }
        ],
        "max_tokens": 1,
        "stream": True,
    },
    "long_stream": {
        "messages": [
            {
                "role": "user",
                "content": "Output every integer from 1 through 400, separated by commas. Do not summarize or use ellipses. Begin immediately with 1.",
            }
        ],
        "max_tokens": 512,
        "stream": True,
    },
    "sampler_topk1": {
        "messages": [
            {
                "role": "user",
                "content": "Choose a color for a fictional spaceship. Reply with the color only.",
            }
        ],
        "max_tokens": 1,
        "stream": True,
        "parameters": {"top_k": 1},
    },
    "sampler_nucleus": {
        "messages": [
            {
                "role": "user",
                "content": "Choose a color for a fictional spaceship. Reply with the color only.",
            }
        ],
        "max_tokens": 1,
        "stream": True,
        "parameters": {"top_p": 0.2},
    },
    "conversation_replay": {
        "messages": [
            {"role": "user", "content": "Remember this secret: cedar-731."},
            {"role": "assistant", "content": "I will remember cedar-731."},
            {"role": "user", "content": "What secret did I give you?"},
        ],
        "max_tokens": 64,
        "stream": True,
    },
}


def expand_probe(probe):
    import copy

    recipe = probe.get("recipe")
    if recipe is not None and recipe not in RECIPES:
        raise ValueError(f"unknown recipe: {recipe}")
    result = copy.deepcopy(RECIPES.get(recipe, {}))
    result.update({k: v for k, v in probe.items() if k != "recipe"})
    return result
