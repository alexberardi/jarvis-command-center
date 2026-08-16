"""Build a ``[turn context: ...]`` line from the node's turn provenance.

Two very different situations open the mic, and they have opposite
failure costs for the ``<not_for_me/>`` decision:

* **Fresh wake** — the user said the wake word. The wake word itself is
  the addressing signal; OWW's detection score says how clean the fire
  was. A false silence here means the user summoned Jarvis by name and
  got ignored — the worst outcome the system can produce (2026-07-25
  prod incident: "what's my son's name?" → ``<not_for_me/>`` despite the
  memory existing and the recall tool being offered).
* **Follow-up window** — no wake word at all. The node keeps the mic
  open after TTS to catch a continuation of the exchange. A false answer
  here means Jarvis butts into the room's resumed conversation and keeps
  going; a false silence costs one re-wake.

``NOT_FOR_ME_INSTRUCTION`` (in the cached system-prompt prefix) defines a
per-mode posture; this hint line is what selects the mode for the turn.
It rides on the user message — like the direction hint — so the cached
prefix stays byte-identical across turns and the warmup KV cache is
never invalidated.

Older node clients don't send ``turn_source``. Only the wake path
measures ``pre_wake_speech_seconds``, so its presence implies a fresh
wake; with neither signal we emit nothing and behavior is unchanged.
"""

from __future__ import annotations

from app.core.direction_hint import BORDERLINE_CONFIDENCE, QUIET_THRESHOLD_S
from app.core.transcript_filter import is_device_command_shaped

# OWW detection scores at or above this are treated as a clean, deliberate
# wake ("the user said your wake word"). Below it, the wake may be a false
# fire and the transcript itself has to carry the addressing evidence.
# Prod wakes on real requests routinely score 0.9+; marginal/false fires
# cluster near the node's trigger threshold (~0.5).
# Alias of direction_hint.BORDERLINE_CONFIDENCE — the single source for the
# confident/marginal wake-score boundary (they were duplicated 0.75s kept
# in sync by hand before).
WAKE_CONFIDENT_THRESHOLD: float = BORDERLINE_CONFIDENCE

# From this follow-up iteration onward the window has been open long enough
# that the room's conversation has likely resumed — require explicit
# engagement instead of inferring continuation.
FOLLOW_UP_STRICT_ITERATION: int = 3


def build_turn_hint(
    turn_source: str | None,
    wake_confidence: float | None = None,
    follow_up_iteration: int | None = None,
    pre_wake_speech_seconds: float | None = None,
    wake_verified: bool | None = None,
    transcript: str | None = None,
) -> str | None:
    """Return a one-line ``[turn context: ...]`` hint, or None.

    Args:
        turn_source: ``"wake"`` or ``"follow_up"`` when the node reports
            how the mic came to be open. Unrecognized values degrade to
            the inference path (treated as absent).
        wake_confidence: OWW detection score for the wake fire (0-1).
        follow_up_iteration: 1-based iteration of the follow-up window.
        pre_wake_speech_seconds: Pre-wake VAD signal — used only to infer
            wake mode for old clients that don't send ``turn_source``.
        wake_verified: Wake-clip verification verdict (bias mode). False
            selects a mild misfire-leaning posture — unless the transcript
            is device-command-shaped (see ``transcript``).
        transcript: The command transcript, when available. A device-
            command-shaped transcript ("turn on the ... lights") keeps the
            normal directed posture even when ``wake_verified`` is False —
            acoustic-side evidence alone must not produce a suppression-
            leaning hint on an imperative device command.

    Returns:
        The bracketed hint to append to the user message, or ``None``
        when there is no provenance signal at all (old client, alternate
        entry path) — in which case behavior is identical to before this
        module existed.
    """
    if turn_source == "follow_up":
        return _follow_up_hint(follow_up_iteration)
    if turn_source == "wake":
        return _wake_hint(wake_confidence, wake_verified, transcript)
    if turn_source == "chat":
        # Typed into the app — there is no microphone in this loop, so
        # "overheard speech" is impossible by construction.
        return (
            "[turn context: typed message — the user wrote this directly "
            "to you in the app. It cannot be overheard speech; "
            "<not_for_me/> does not apply. Answer it or call the right "
            "tool.]"
        )
    # No (recognized) source: only the wake path measures pre-wake VAD,
    # so a reported value implies a fresh wake. Never claim a confidence
    # we weren't given.
    if pre_wake_speech_seconds is not None:
        return _wake_hint(None)
    return None


def should_double_check_sentinel(
    turn_source: str | None,
    wake_confidence: float | None = None,
    follow_up_iteration: int | None = None,
    pre_wake_speech_seconds: float | None = None,
) -> bool:
    """Should a first-look ``<not_for_me/>`` buy a reasoned second opinion?

    Only when the acoustics clearly say "directed at Jarvis" — a
    quiet-room wake or a high-confidence OWW fire. On those turns a snap
    sentinel from the no-think model is more likely a pattern-match miss
    (2026-07-27: "who is Leo?" silenced because "Leo" is a pet name) than
    a real addressing verdict, and one /think re-check is cheap next to a
    false silence. Ambient and middle-band turns stay single-pass so
    genuine overheard-speech rejections aren't argued with. EARLY follow-up
    iterations (< FOLLOW_UP_STRICT_ITERATION) also qualify: a tool-needing
    continuation ("...actually set a timer for 5 minutes") must not be
    snap-silenced by a first-look ``<not_for_me/>``. Late follow-up iterations
    stay single-pass — by then silence is the window's designed ending.

    An unverified wake clip deliberately does NOT disable the rescue: the
    verify clip itself can be garbage (prod 2026-08-15: two real lights
    commands were suppressed because verification read junk clips), so one
    bad clip must never single-handedly silence a turn. The verdict is a
    soft bias in the wake hint, and the /think re-check stays available to
    catch exactly that failure.
    """
    if turn_source == "follow_up":
        # Early iterations get the /think rescue — a genuine continuation may
        # need a tool and must not be dropped by a snap sentinel. Late
        # iterations stay strict, where going quiet is the designed ending.
        return (follow_up_iteration or 1) < FOLLOW_UP_STRICT_ITERATION
    # A typed app message is directed at Jarvis by construction — any
    # sentinel on it deserves the reasoned re-check.
    if turn_source == "chat":
        return True
    if (
        turn_source == "wake"
        and wake_confidence is not None
        and wake_confidence >= WAKE_CONFIDENT_THRESHOLD
    ):
        return True
    # Quiet room right before the wake — the direction hint's "strong
    # signal this is directed at you" band.
    return (
        pre_wake_speech_seconds is not None
        and pre_wake_speech_seconds < QUIET_THRESHOLD_S
    )


def _wake_hint(
    wake_confidence: float | None,
    wake_verified: bool | None = None,
    transcript: str | None = None,
) -> str:
    # Wake-clip verification verdict: the clip that FIRED the wake was
    # transcribed and contained nothing wake-word-shaped. That is ONE
    # signal, not a verdict — the verify clip itself can be garbage (prod
    # 2026-08-15: two real "turn on the ... lights" commands were suppressed
    # by unreadable clips), so the posture is a MILD ambient lean, never a
    # suppression instruction. Only bias mode reaches here; enforce
    # short-circuits upstream. The imperative device-command guard is
    # senior: a device-shaped transcript keeps the normal directed posture
    # because acoustic-side evidence alone must not talk the model out of
    # an imperative command.
    if wake_verified is False and not (
        transcript is not None and is_device_command_shaped(transcript)
    ):
        return (
            "[turn context: the recorded wake clip did not clearly contain "
            "the wake word when transcribed — weigh this as one signal that "
            "the wake may have been a detector misfire, not as proof. A "
            "coherent command or question addressed to you should still be "
            "answered. Emit <not_for_me/> only when the transcript ALSO "
            "reads like speech meant for someone else (a reply to another "
            "person, a mid-story line, a dialogue fragment).]"
        )
    if wake_confidence is not None and wake_confidence < WAKE_CONFIDENT_THRESHOLD:
        return (
            f"[turn context: wake word fired at low confidence "
            f"({wake_confidence:.2f}) — possibly a false wake. Judge by the "
            f"transcript: a coherent command or question is still for you; "
            f"fragments, half-sentences, or noise are not.]"
        )
    scored = (
        f" (detection confidence {wake_confidence:.2f})"
        if wake_confidence is not None
        else ""
    )
    return (
        # "answer DIRECTLY from your User Profile … ONLY call recall when NOT
        # already in your profile" — paired with the build_speaker_block change,
        # this stops the model firing a redundant `recall` round-trip for a fact
        # that's already in the injected profile (0/6 recall in a 12-tool A/B vs
        # ~50% with the old "call recall — do not go silent" phrasing). The
        # redundant recall doubled latency (~2.2s prefill/call on the cacheless 9B).
        #
        # The "if it merely ASKS … but if the user REPORTS/REQUESTS an action,
        # call the tool" carve-out is load-bearing too: without it, an action
        # statement whose subject is in the profile ("I gave Leo his medicine" —
        # profile has "Administers Keppra to Leo") reads as a profile topic, and
        # the 9B narrates a fake confirmation instead of calling the tool
        # (prod 2026-08). Verbatim-prompt A/B vs the prod 9B: 1–2/8 tool calls
        # without this line, 8/8 with it (+ the mandate/speaker carve-outs);
        # "Who is Leo?" still answers directly 8/8 (no recall regression).
        f"[turn context: fresh wake — the user said your wake word{scored}. "
        f"This turn is addressed to you: answer it or run the tool it calls "
        f"for. If it merely ASKS about the speaker's own life, answer DIRECTLY "
        f"from your User Profile when the fact is listed there (only call recall "
        f"when it's NOT — never go silent); but if the user REPORTS or REQUESTS "
        f"an action, call the tool that does it — never just say you did it. "
        f"Reserve <not_for_me/> for STT artifacts or speech explicitly aimed at "
        f"another person.]"
    )


def _follow_up_hint(follow_up_iteration: int | None) -> str:
    iteration = max(1, follow_up_iteration or 1)
    escalation = (
        " This window has stayed open across several turns; the room has "
        "likely moved on — respond only to explicit engagement (your name, "
        "a direct question to you, an unmistakable continuation)."
        if iteration >= FOLLOW_UP_STRICT_ITERATION
        else ""
    )
    return (
        f"[turn context: follow-up window, iteration {iteration} — there "
        f"was no wake word; the mic stayed open after your last reply to "
        f"catch a continuation. If this clearly continues your exchange, "
        f"answer it — or run the tool it calls for; don't reply with bare "
        f"prose when the request needs a tool. If the room's conversation has "
        f"moved on without you, emit <not_for_me/> — your exchange is over, "
        f"and going quiet is the designed ending, not a failure.{escalation}]"
    )
