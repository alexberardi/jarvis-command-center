"""The situation prompt groups the ACTION menu by which commands are DESIGNED
for the signal kinds actually present (IJarvisCommand.listening_signal_types),
vs the general pool. Sharpens proactive-proposal precision."""
from app.services.proposal_matcher import _build_situation_prompt


def _menu(cmd, action, listening):
    return {
        "command": cmd, "action": action, "description": f"{cmd} {action}",
        "args": {}, "listening_signal_types": listening,
    }


def test_segments_designed_before_general():
    bundle = [{"source_key": "appt:1", "kind": "appt.detected", "data": {"title": "x"}}]
    menu = [
        _menu("play_music", "play", []),                       # general
        _menu("add_event", "create_event", ["appt.detected"]),  # designed for the present kind
    ]
    prompt = _build_situation_prompt(bundle, menu)
    assert "ACTIONS designed for the current signals:" in prompt   # the segmented header
    assert "Other available actions" in prompt
    # the purpose-built command is listed in the designed section, ahead of the general pool
    assert prompt.index("add_event") < prompt.index("play_music")


def test_no_match_means_single_general_section():
    bundle = [{"source_key": "w", "kind": "weather", "data": {}}]
    menu = [_menu("add_event", "create_event", ["appt.detected"])]
    prompt = _build_situation_prompt(bundle, menu)
    assert "ACTIONS designed for the current signals:" not in prompt   # nothing designed
    assert "add_event" in prompt                                        # still offered


def test_kindless_bundle_is_backward_compatible():
    # the match_proposals adapter passes {source_key, data} with no kind.
    bundle = [{"source_key": None, "data": {"title": "x"}}]
    menu = [_menu("add_event", "create_event", ["appt.detected"])]
    prompt = _build_situation_prompt(bundle, menu)
    assert "ACTIONS designed for the current signals:" not in prompt
    assert "add_event" in prompt
