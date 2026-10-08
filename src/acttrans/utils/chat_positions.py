"""Template-relative candidate token positions for masked-diffusion LM steering
extraction (arXiv 2512.24143v2, "Activation Steering for Masked Diffusion
Language Models").

Candidate positions are defined RELATIVE TO THE CHAT TEMPLATE STRUCTURE rather
than as a fixed offset like -1/"mean"/"mask": a few tokens at the very start of
the templated prompt (pre-instruction — before the user's question even
begins) and a few tokens right where the user's turn ends and the assistant
turn opens (post-instruction — the existing appended-[MASK] readout used
elsewhere in this repo is one member of this set). This only matters for a
model whose attention is bidirectional (the two LLaDA families): in a causal
LM the pre-instruction tokens can never see the user's question, so there is
nothing to extract there — which is exactly the paper's point.
"""

# Two placeholders of different length AND disjoint character content, so the
# only tokens the two templated texts can possibly share are the fixed
# boilerplate before/after the user turn — a coincidental token match inside
# the placeholder content itself would corrupt the boundary detection below.
_SHORT_PLACEHOLDER = "aaaa"
_LONG_PLACEHOLDER = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _template_ids(tok, content: str):
    text = tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    return tok(text, add_special_tokens=False)["input_ids"]


def find_template_boundaries(tok, n_pre: int = 4, n_post: int = 4):
    """(prefix_len, suffix_len): the count of leading / trailing template
    tokens that do NOT depend on the user's question content, found as the
    longest common prefix / suffix of the token-id sequences for two
    differently-sized placeholder questions. Both are capped at n_pre/n_post —
    only a handful of structural delimiter tokens are wanted as candidates,
    not the entire boilerplate."""
    ids_short = _template_ids(tok, _SHORT_PLACEHOLDER)
    ids_long = _template_ids(tok, _LONG_PLACEHOLDER)

    prefix_len = 0
    for a, b in zip(ids_short, ids_long):
        if a != b:
            break
        prefix_len += 1

    suffix_len = 0
    for a, b in zip(reversed(ids_short), reversed(ids_long)):
        if a != b:
            break
        suffix_len += 1
    # Clamp so prefix+suffix never exceeds the shorter sequence — otherwise a
    # short placeholder collapsing to very few tokens could have its "prefix"
    # and "suffix" regions overlap and double-count the same token.
    suffix_len = min(suffix_len, len(ids_short) - prefix_len)

    return min(prefix_len, n_pre), min(suffix_len, n_post)


def candidate_positions(tok, question: str, n_pre: int = 4, n_post: int = 4):
    """Absolute token indices, within THIS item's template_text (the
    chat-templated question WITHOUT the appended "Answer: {choice}"), that are
    stable structural markers:

      pre  -- counts forward from sequence start (pre-instruction)
      post -- counts backward from the end of template_text, i.e. from the
              point where "Answer: " will be appended (post-instruction)

    Returns (template_text, template_len, pre_positions, post_positions) so
    callers needing the tokenized prompt don't have to rebuild it.
    """
    prefix_len, suffix_len = find_template_boundaries(tok, n_pre, n_post)
    template_text = tok.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    template_len = len(tok(template_text, add_special_tokens=False)["input_ids"])

    pre_positions = list(range(prefix_len))
    post_positions = list(range(template_len - suffix_len, template_len))
    return template_text, template_len, pre_positions, post_positions


def assert_positions_stable(tok, questions, n_pre: int = 4, n_post: int = 4):
    """Self-check: the token IDENTITY at every pre-instruction offset, and at
    every post-instruction offset counted backward, must match across every
    item in `questions` — a "structural marker" that silently drifts with
    content length would mix unrelated positions together across items, which
    is exactly the failure mode the paper's own position taxonomy is meant to
    avoid. Raises with the offending question and token ids on drift."""
    ref = None
    for q in questions:
        template_text, _, pre, post = candidate_positions(tok, q, n_pre, n_post)
        ids = tok(template_text, add_special_tokens=False)["input_ids"]
        pre_tokens = tuple(ids[i] for i in pre)
        post_tokens = tuple(ids[i] for i in post)
        if ref is None:
            ref = (pre_tokens, post_tokens)
            continue
        if (pre_tokens, post_tokens) != ref:
            raise ValueError(
                f"Template-relative positions drifted for question {q!r}: "
                f"pre {pre_tokens} vs reference {ref[0]}, "
                f"post {post_tokens} vs reference {ref[1]}."
            )
