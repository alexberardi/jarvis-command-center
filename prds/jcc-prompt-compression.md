# JCC Prompt Compression (PRD)

## Goal

Reduce the warmup/system prompt size so the system performs reliably within ~4k context windows on small models, while preserving tool accuracy and multi-turn capability.

## Motivation

Current prompt size is too large for small-context LLMs. We need strategies to shrink what is sent to the model without removing critical guidance or weakening tool selection.

## Scope

This PRD proposes three complementary strategies:

1. **Server-side command filtering** before prompt build.
2. **Category-based narrowing** via a server tool.
3. **Move anti-patterns out of warmup prompt** into a tool-lookup flow.

## Strategy 1: Server-Side Command Filtering (Fuzzy Match)

### Summary

Before building the prompt, the server trims the list of available commands based on how well the user’s voice command matches the command’s keywords (and optional examples).

### Expected Behavior (Plain Language)

- The server looks at the user’s voice command and the keywords attached to each command.
- It keeps only commands that “seem related” to the user’s phrasing.
- If nothing clearly matches, it falls back to the full command list or a small default set.

### Intended Outcomes

- Fewer tools in the prompt.
- Faster tool selection.
- Smaller model can focus on relevant tools instead of reading everything.

### Constraints

- Must never drop “critical” tools (e.g., resolve_relative_date, request_validation).
- Must include any tools explicitly referenced by the user (if a tool name appears in the text).

### Alex's Notes
- Yes. So server side tools should not be included in the filtering logic. They should always all be available.

## Strategy 2: Category-Based Narrowing

### Summary

Each command has a **client-specified category**. The model first chooses a category using a new server tool, then the server provides only the commands from that category.

### Required Additions

- Add a `category` field to each command (provided by the client).
- Add a new server tool: `category_search`.

### category_search Tool (High-Level Shape)

**Input**

- `voice_command` (string)
- `available_categories` (array of strings)

**Output**

- `selected_category` (string)
- `confidence` (optional)

### Intended Flow

1. Warmup prompt includes only categories + the tool to select one.
2. The model calls `category_search`.
3. Server responds with the selected category.
4. Prompt is rebuilt with only commands in that category.

### Notes

- Categories are client-controlled to align with product semantics.
- Categories should be short, human-readable, and stable.

## Strategy 3: Move Anti-Patterns Out of Warmup Prompt

### Summary

Remove anti-patterns from the main prompt. Provide them only when the model requests examples or disambiguation.

### Options

- Add a new server tool: `get_command_antipatterns`
  - Returns anti-patterns for the requested command(s).
- Or, include anti-patterns in the existing `get_command_utterance_examples` tool response.

### Intended Outcomes

- Reduced prompt size on every request.
- Anti-patterns only appear when the model is unsure.

## Success Criteria

- Warmup prompt fits within ~4k context window.
- No regression in tool-call accuracy compared to current baseline.
- Smaller models (1B–3B) maintain ≥90% test pass rate.

## Open Questions

- Should command filtering run on **every** user turn, or only at warmup?
    - That's a great question. I think for now, since we're not truly conversational yet anyway, we should just do it on warmup.
- What should happen if the category tool selects “wrong” category?
    - What do you mean by wrong? Like it selects "Math" which exists but should've chosen "General Knowledge" or something? Or do you mean it selected a category that doesn't exist? I think for both, we should make it obvious that the category_search command can be run multiple times. heck, we can allow up to X runs based on how many categories exist. If the category supplied doesn't exist, the implementation should reprompt that it was an invalid selection and re-provide the correct available categories. Does that answer your question?
- Should anti-patterns be bundled with examples or exposed separately?
    - I'm torn. I think if they're separate it'll be more likely to be skipped. So for now, let's include in the examples even though it will inflate the token size of that tool.

## Non-Goals

- No changes to the client tool execution loop itself.
- No changes to command schemas beyond adding optional `category`.
