#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Prompt Templates (v2)
=============================================
Includes COMBINED prompts that generate Target + Intent + Implication
in a single LLM call (3x faster than separate calls).

Variants: vanilla, few_shot, cot — each with per-field and combined modes.
"""


# =============================================================================
# Shared intro
# =============================================================================
SYSTEM_INTRO = (
    "You are an expert in analyzing hateful comments driven by fake narratives.\n"
    "Based on the given comment, provide the requested analysis."
)

FIELD_INSTRUCTIONS = {
    "Target": (
        "Identify the *Target* (who is being targeted) in the following comment.\n"
        "If there is one target, only mention that. If there are multiple, "
        "mention each target as a single word and separate them by commas.\n"
        "Respond only with the target(s); do not provide any explanation."
    ),
    "Intent": (
        "Briefly describe the *Intent* (motive or purpose behind the comment) "
        "using a single concise sentence."
    ),
    "Implication": (
        "Briefly describe the possible *Implication* (impact or consequence on society) "
        "in a single concise sentence."
    ),
}


# =============================================================================
# COMBINED prompt — all 3 fields in ONE generation (3x faster)
# =============================================================================
COMBINED_INSTRUCTION = """Analyze the following comment and provide:

1. **Target**: Who is being targeted? If one target, state it as a single word. If multiple, list each as a single word separated by commas.
2. **Intent**: What is the motive or purpose behind this comment? Answer in one concise sentence.
3. **Implication**: What is the potential social impact or consequence? Answer in one concise sentence.

Respond EXACTLY in this format (one line each, no extra text):
Target: <your answer>
Intent: <your answer>
Implication: <your answer>"""


def combined_vanilla_prompt(comment: str, context: str = "", **kwargs) -> str:
    """Combined prompt: all 3 fields in one call."""
    ctx_block = ""
    if context:
        ctx_block = (
            f"\nThe following context provides background information:\n"
            f"[Context]: \"{context}\"\n"
        )
    return (
        f"{SYSTEM_INTRO}\n{ctx_block}\n"
        f"{COMBINED_INSTRUCTION}\n\n"
        f"## Comment: \"{comment}\"\n\n"
        f"Target:"
    )


def combined_few_shot_prompt(comment: str, examples: list = None,
                              context: str = "", **kwargs) -> str:
    """Combined few-shot: examples + all 3 fields in one call."""
    ctx_block = ""
    if context:
        ctx_block = (
            f"\nThe following context provides background information:\n"
            f"[Context]: \"{context}\"\n"
        )

    examples_text = ""
    if examples:
        for i, ex in enumerate(examples, 1):
            examples_text += (
                f"### Example {i}\n"
                f"Comment: \"{ex['comment']}\"\n"
                f"Target: {ex['Target']}\n"
                f"Intent: {ex['Intent']}\n"
                f"Implication: {ex['Implication']}\n\n"
            )

    return (
        f"{SYSTEM_INTRO}\n{ctx_block}\n"
        f"{COMBINED_INSTRUCTION}\n\n"
        f"{examples_text}"
        f"Now analyze:\n\n"
        f"## Comment: \"{comment}\"\n\n"
        f"Target:"
    )


def combined_cot_prompt(comment: str, context: str = "", **kwargs) -> str:
    """Combined chain-of-thought: reason then output all 3 fields."""
    ctx_block = ""
    if context:
        ctx_block = (
            f"\nThe following context provides background information:\n"
            f"[Context]: \"{context}\"\n"
        )
    return (
        f"{SYSTEM_INTRO}\n{ctx_block}\n"
        f"Think step by step:\n"
        f"Step 1: Identify the fake claim or misinformation the comment relies on.\n"
        f"Step 2: Determine who is being targeted.\n"
        f"Step 3: Consider the commenter's motivation.\n"
        f"Step 4: Consider the potential social impact.\n\n"
        f"Then respond EXACTLY in this format (one line each):\n"
        f"Target: <your answer>\n"
        f"Intent: <your answer>\n"
        f"Implication: <your answer>\n\n"
        f"## Comment: \"{comment}\"\n\n"
        f"Target:"
    )


# =============================================================================
# Per-field prompts (original — kept for compatibility)
# =============================================================================
def vanilla_prompt(comment: str, field: str, **kwargs) -> str:
    instruction = FIELD_INSTRUCTIONS[field]
    return (
        f"{SYSTEM_INTRO}\n\n"
        f"{instruction}\n\n"
        f"## Comment: \"{comment}\"\n"
        f"## {field}:"
    )


def few_shot_prompt(comment: str, field: str, examples: list = None, **kwargs) -> str:
    if not examples:
        return vanilla_prompt(comment, field)
    instruction = FIELD_INSTRUCTIONS[field]
    examples_text = ""
    for i, ex in enumerate(examples, 1):
        examples_text += (
            f"### Example {i}\n"
            f"## Comment: \"{ex['comment']}\"\n"
            f"## {field}: {ex[field]}\n\n"
        )
    return (
        f"{SYSTEM_INTRO}\n\n"
        f"{instruction}\n\n"
        f"Here are some examples:\n\n"
        f"{examples_text}"
        f"Now analyze the following comment:\n\n"
        f"## Comment: \"{comment}\"\n"
        f"## {field}:"
    )


def cot_prompt(comment: str, field: str, **kwargs) -> str:
    COT_INSTRUCTIONS = {
        "Target": "Step 1: Identify the fake claim.\nStep 2: Who is targeted?\nStep 3: State only the target(s).\n\nProvide ONLY the final target(s).",
        "Intent": "Step 1: Identify the fake claim.\nStep 2: What is the commenter trying to achieve?\nStep 3: Describe intent in one sentence.\n\nProvide ONLY the final intent.",
        "Implication": "Step 1: Identify the fake claim.\nStep 2: What social impact could this have?\nStep 3: Describe implication in one sentence.\n\nProvide ONLY the final implication.",
    }
    return (
        f"{SYSTEM_INTRO}\n\n"
        f"Think step by step:\n{COT_INSTRUCTIONS[field]}\n\n"
        f"## Comment: \"{comment}\"\n"
        f"## {field}:"
    )


def rag_prompt(comment: str, field: str, context: str = "",
               base_variant: str = "vanilla", **kwargs) -> str:
    context_block = (
        f"The following context provides background information:\n"
        f"[Context]: \"{context}\"\n\n"
        f"Use this context to ground your analysis.\n\n"
    )
    instruction = FIELD_INSTRUCTIONS[field]
    return (
        f"{SYSTEM_INTRO}\n\n"
        f"{context_block}"
        f"{instruction}\n\n"
        f"## Comment: \"{comment}\"\n"
        f"## {field}:"
    )


# =============================================================================
# Registries
# =============================================================================
PROMPT_REGISTRY = {
    "vanilla": vanilla_prompt,
    "few_shot": few_shot_prompt,
    "cot": cot_prompt,
    "rag": rag_prompt,
}

COMBINED_PROMPT_REGISTRY = {
    "vanilla": combined_vanilla_prompt,
    "few_shot": combined_few_shot_prompt,
    "cot": combined_cot_prompt,
}


def get_prompt(variant: str, comment: str, field: str, **kwargs) -> str:
    """Get a rendered per-field prompt."""
    if variant not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[variant](comment, field, **kwargs)


def get_combined_prompt(variant: str, comment: str, **kwargs) -> str:
    """Get a rendered combined prompt (all 3 fields in one call)."""
    base = variant.replace("rag_", "")
    if base not in COMBINED_PROMPT_REGISTRY:
        base = "vanilla"
    return COMBINED_PROMPT_REGISTRY[base](comment, **kwargs)
