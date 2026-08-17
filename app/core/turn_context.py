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

from app.core.direction_hint import (
    BORDERLINE_CONFIDENCE,
    QUIET_THRESHOLD_S,
    is_media_self_playback,
)
from app.core.transcript_filter import (
    is_device_command_shaped,
    is_music_control_shaped,
)

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

# Verdict-propagation sentinel value: the conversation's WAKE turn resolved
# an ``unverified`` wake-clip verdict (app/core/wake_verification.py). Only
# this exact value selects the doubted follow-up posture — ``verified`` is
# trusted engagement, and ``clip_unreliable``/absent is NO-SIGNAL, not doubt
# (the verify clip itself can be garbage; fail open).
DOUBTED_WAKE_VERDICT: str = "unverified"


def build_turn_hint(
    turn_source: str | None,
    wake_confidence: float | None = None,
    follow_up_iteration: int | None = None,
    pre_wake_speech_seconds: float | None = None,
    wake_verified: bool | None = None,
    transcript: str | None = None,
    self_playback: bool | None = None,
    self_playback_kind: str | None = None,
    conversation_wake_verdict: str | None = None,
    doubt_round: int | None = None,
    doubt_max_rounds: int | None = None,
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
        self_playback: The node's own speaker was playing media when the
            wake fired. Adds a mild media-context line to the wake hint
            (music is both a false-wake source AND something users talk
            over to control), and music-control-shaped transcripts gain
            the device-command guard's seniority.
        self_playback_kind: What was playing ("music"). See
            ``direction_hint.is_media_self_playback``.
        conversation_wake_verdict: The wake-clip verdict of the WAKE turn
            that opened this conversation, propagated onto follow-up turns
            (which have no wake clip of their own to re-check). Only
            ``"unverified"`` (DOUBTED_WAKE_VERDICT) changes anything: the
            follow-up hint becomes a caution posture — the conversation
            began as a suspected detector misfire, so room-conversation-
            shaped utterances should be sentineled. ``verified`` /
            ``clip_unreliable`` / None keep the hint byte-identical to
            today (fail open: clip_unreliable and absent are no-signal,
            not doubt). Ignored on non-follow-up turns.
        doubt_round: How many answered (non-sentinel) rounds this
            conversation has already had, for the doubted-conversation
            round cap. Only meaningful with an unverified verdict.
        doubt_max_rounds: The ``voice.followup_doubt_max_rounds`` setting.
            When a DOUBTED conversation has had ``doubt_round >=
            doubt_max_rounds`` answered rounds, the caution hint gains a
            strong wrap-up lean (answer briefly and close, or sentinel) —
            a lean, never a hard cut. Verified conversations are exempt.

    Returns:
        The bracketed hint to append to the user message, or ``None``
        when there is no provenance signal at all (old client, alternate
        entry path) — in which case behavior is identical to before this
        module existed.
    """
    media_playback = is_media_self_playback(self_playback, self_playback_kind)
    if turn_source == "follow_up":
        return _follow_up_hint(
            follow_up_iteration,
            wake_verdict=conversation_wake_verdict,
            doubt_round=doubt_round,
            doubt_max_rounds=doubt_max_rounds,
            transcript=transcript,
            media_playback=media_playback,
        )
    if turn_source == "wake":
        return _wake_hint(
            wake_confidence, wake_verified, transcript, media_playback
        )
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
        return _wake_hint(None, media_playback=media_playback)
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
    catch exactly that failure. The same holds for DOUBTED-conversation
    follow-ups (conversation_wake_verdict="unverified"): this gate never
    consults the verdict, so early follow-up iterations keep the rescue
    even when the caution hint leans toward the sentinel — fail-open, a
    real engaged user must not fight suppression.
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


# Mild media-context line appended (inside the bracket) to every wake-hint
# variant when the node reported self-playback. Deliberately two-sided:
# background media speech IS a known false-wake source, but users also
# routinely talk to Jarvis OVER their music — most often to control it —
# so the line weighs the transcript rather than leaning either way.
_MEDIA_CONTEXT_NOTE = (
    " Context: music was playing from this node's own speaker when the wake "
    "fired. Speech or lyrics in the media can false-wake the node, but "
    "people also often talk to you over their music — especially to change "
    "or stop it — so weigh the transcript content itself. A music-control "
    "command (stop, pause, skip, next, play, volume) is directed at you: "
    "act on it."
)


def _with_media_note(hint: str, media_playback: bool) -> str:
    """Append the media-context note inside the hint's closing bracket."""
    if not media_playback:
        return hint
    return hint[:-1] + _MEDIA_CONTEXT_NOTE + "]"


def _wake_hint(
    wake_confidence: float | None,
    wake_verified: bool | None = None,
    transcript: str | None = None,
    media_playback: bool = False,
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
    # an imperative command. During self-playback, music-control shapes
    # ("pause", "skip") share that seniority — with residual music bleed in
    # the wake clip, an unverified verdict is expected on a REAL mid-music
    # wake, and controlling the music is the most likely thing being said.
    if wake_verified is False and not (
        transcript is not None
        and (
            is_device_command_shaped(transcript)
            or (media_playback and is_music_control_shaped(transcript))
        )
    ):
        return _with_media_note(
            "[turn context: the recorded wake clip did not clearly contain "
            "the wake word when transcribed — weigh this as one signal that "
            "the wake may have been a detector misfire, not as proof. A "
            "coherent command or question addressed to you should still be "
            "answered. Emit <not_for_me/> only when the transcript ALSO "
            "reads like speech meant for someone else (a reply to another "
            "person, a mid-story line, a dialogue fragment).]",
            media_playback,
        )
    if wake_confidence is not None and wake_confidence < WAKE_CONFIDENT_THRESHOLD:
        return _with_media_note(
            f"[turn context: wake word fired at low confidence "
            f"({wake_confidence:.2f}) — possibly a false wake. Judge by the "
            f"transcript: a coherent command or question is still for you; "
            f"fragments, half-sentences, or noise are not.]",
            media_playback,
        )
    scored = (
        f" (detection confidence {wake_confidence:.2f})"
        if wake_confidence is not None
        else ""
    )
    return _with_media_note(
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
        f"another person.]",
        media_playback,
    )


def _follow_up_hint(
    follow_up_iteration: int | None,
    wake_verdict: str | None = None,
    doubt_round: int | None = None,
    doubt_max_rounds: int | None = None,
    transcript: str | None = None,
    media_playback: bool = False,
) -> str:
    iteration = max(1, follow_up_iteration or 1)
    # Verdict propagation (the kitchen runaway, 2026-08-15): a false wake
    # that survives the wake-turn hints and gets ANSWERED opens the
    # follow-up window onto the family's continuing conversation — and
    # follow-up turns have no wake clip to re-verify, so without carrying
    # the WAKE verdict forward every follow-up gets the engaged-
    # conversation posture and the window re-opens after each answer.
    # Only an ``unverified`` verdict selects the caution posture;
    # verified / clip_unreliable / absent keep this hint byte-identical
    # to before the verdict propagated (fail open). The imperative
    # device-command guard stays senior — same shape as the wake hint:
    # a device-shaped transcript (or a music-control shape during
    # self-playback) keeps the normal directed posture, because a REAL
    # engaged user must never have to fight suppression.
    doubted = wake_verdict == DOUBTED_WAKE_VERDICT and not (
        transcript is not None
        and (
            is_device_command_shaped(transcript)
            or (media_playback and is_music_control_shaped(transcript))
        )
    )
    if doubted:
        # Round cap: a LEAN toward closing, never a hard cut — the model
        # still answers (briefly) when the utterance is addressed to it.
        wrap_up = (
            (
                f" This suspected-misfire conversation has already run "
                f"{doubt_round} answered rounds — wrap up now: if this "
                f"really is addressed to you, answer briefly and append "
                f"<exchange_complete/> to close the exchange; if it is "
                f"not, emit <not_for_me/>."
            )
            if (
                doubt_round is not None
                and doubt_max_rounds is not None
                and doubt_round >= doubt_max_rounds
            )
            else ""
        )
        return (
            f"[turn context: follow-up window, iteration {iteration} — "
            f"there was no wake word, and this conversation BEGAN as a "
            f"suspected detector misfire (the original wake clip did not "
            f"contain the wake word when transcribed). The mic stayed open "
            f"after your last reply, so what it hears now is likely the "
            f"room's own conversation continuing without you. If this "
            f"utterance reads like people talking to each other (a reply "
            f"to someone else, a third-person reference, a mid-story line, "
            f"a plan being made between people), emit <not_for_me/>. If it "
            f"is unmistakably addressed to you, answer it — or run the "
            f"tool it calls for.{wrap_up}]"
        )
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
