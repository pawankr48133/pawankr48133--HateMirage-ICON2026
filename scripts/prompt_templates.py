#!/usr/bin/env python3
"""
HateMirage ICON 2026 — Prompt Templates
========================================
Centralized prompt templates for all variants:
  - vanilla   : direct zero-shot (matches starter kit prompt)
  - few_shot  : prepend N examples from training data
  - cot       : chain-of-thought reasoning

Each template function takes (comment, field, **kwargs) and returns
a fully rendered prompt string.
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
# Vanilla (Zero-Shot) — matches the starter kit prompt
# =============================================================================
def vanilla_prompt(comment: str, field: str, **kwargs) -> str:
    """Standard zero-shot prompt from the starter kit."""
    instruction = FIELD_INSTRUCTIONS[field]
    return (
        f"{SYSTEM_INTRO}\n\n"
        f"{instruction}\n\n"
        f"## Comment: \"{comment}\"\n"
        f"## {field}:"
    )


# =============================================================================
# Few-Shot — prepend labeled examples before the test comment
# =============================================================================
def few_shot_prompt(comment: str, field: str, examples: list[dict] = None, **kwargs) -> str:
    """
    Few-shot prompt with N examples prepended.

    Args:
        examples: list of dicts with keys 'comment', 'Target', 'Intent', 'Implication'
    """
    if not examples:
        # Fall back to vanilla if no examples provided
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


# =============================================================================
# Chain-of-Thought — reason step-by-step
# =============================================================================
COT_INSTRUCTIONS = {
    "Target": (
        "Step 1: Identify the fake claim or misinformation the comment relies on.\n"
        "Step 2: Determine who is being targeted or attacked based on this claim.\n"
        "Step 3: State only the target(s) — one word each, comma-separated if multiple.\n\n"
        "Provide ONLY the final target(s) after your reasoning."
    ),
    "Intent": (
        "Step 1: Identify the fake claim or misinformation the comment relies on.\n"
        "Step 2: Consider what the commenter is trying to achieve or convey.\n"
        "Step 3: Describe the intent in a single concise sentence.\n\n"
        "Provide ONLY the final intent sentence after your reasoning."
    ),
    "Implication": (
        "Step 1: Identify the fake claim or misinformation the comment relies on.\n"
        "Step 2: Consider the potential social impact if this comment were widely spread.\n"
        "Step 3: Describe the implication in a single concise sentence.\n\n"
        "Provide ONLY the final implication sentence after your reasoning."
    ),
}


def cot_prompt(comment: str, field: str, **kwargs) -> str:
    """Chain-of-thought prompt for deeper reasoning."""
    instruction = COT_INSTRUCTIONS[field]
    return (
        f"{SYSTEM_INTRO}\n\n"
        f"Think step by step:\n"
        f"{instruction}\n\n"
        f"## Comment: \"{comment}\"\n"
        f"## {field}:"
    )


# =============================================================================
# RAG-Augmented — injects retrieved context into any base variant
# =============================================================================
def rag_prompt(comment: str, field: str, context: str = "",
               base_variant: str = "vanilla", **kwargs) -> str:
    """
    RAG-augmented prompt: wraps any base variant with retrieved context.

    Args:
        context: concatenated text from top-k retrieved documents
        base_variant: which base prompt style to use ("vanilla", "few_shot", "cot")
    """
    context_block = (
        f"The following context provides background information retrieved from "
        f"credible sources about fake claims and misinformation:\n"
        f"[Context]: \"{context}\"\n\n"
        f"Use this context to ground your analysis.\n\n"
    )

    instruction = FIELD_INSTRUCTIONS[field]
    if base_variant == "cot":
        instruction = COT_INSTRUCTIONS[field]

    return (
        f"{SYSTEM_INTRO}\n\n"
        f"{context_block}"
        f"{instruction}\n\n"
        f"## Comment: \"{comment}\"\n"
        f"## {field}:"
    )


# =============================================================================
# Template dispatcher
# =============================================================================
PROMPT_REGISTRY = {
    "vanilla": vanilla_prompt,
    "few_shot": few_shot_prompt,
    "cot": cot_prompt,
    "rag": rag_prompt,
}


def get_prompt(variant: str, comment: str, field: str, **kwargs) -> str:
    """
    Get a rendered prompt for the given variant.

    Args:
        variant: one of "vanilla", "few_shot", "cot", "rag"
        comment: the input comment text
        field: one of "Target", "Intent", "Implication"
        **kwargs: extra args passed to the template function
            - examples (list[dict]): for few_shot
            - context (str): for rag
            - base_variant (str): for rag (which underlying prompt style)
    """
    if variant not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown prompt variant: {variant}. Choose from {list(PROMPT_REGISTRY.keys())}")
    return PROMPT_REGISTRY[variant](comment, field, **kwargs)
