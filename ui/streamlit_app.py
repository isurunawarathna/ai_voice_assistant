"""Visual harness for the welcome -> language selection -> confirmation flow.

Run the backend, then this page:

    uvicorn app.main:app --port 8000
    streamlit run ui/streamlit_app.py

The page connects to the real ``/api/v1/voice`` WebSocket, so what it shows is
what any client (Angular, telephony gateway) will receive: the same events,
the same audio, the same limits.
"""

from __future__ import annotations

import hashlib
import html
import io
import os
import sys
import wave
from pathlib import Path

import httpx
import streamlit as st

# ``streamlit run ui/streamlit_app.py`` puts ``ui/`` on sys.path, not the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.session_client import (
    AudioItem,
    Entry,
    Snapshot,
    VoiceSessionClient,
    http_base,
)
from ui.view_model import analyse, step_states

DEFAULT_URL = os.getenv("VOICE_WS_URL", "ws://127.0.0.1:8000/api/v1/voice")
REFRESH_SECONDS = 0.7

PROMPT_TITLES = {
    "welcome": "Welcome message",
    "language_retry": "Please choose a language",
    "language_confirm": "Language confirmed",
}
LANGUAGE_NAMES = {"si-LK": "Sinhala", "en-US": "English"}
EXAMPLE_UTTERANCES = [
    ("Sinhala", "Sinhala"),
    ("English", "English"),
    ("ඉංග්ලිෂ්", "ඉංග්ලිෂ්"),
    ("“two”", "two"),
    ("Unclear", "uhh what"),
]

STEPS = [
    ("connect", "Connect", "Socket open, session created"),
    ("welcome", "Welcome", "Greeting audio delivered"),
    ("language", "Language", "Caller picks Sinhala or English"),
    ("ready", "Confirmed", "Confirmation played, ready for questions"),
]

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.vx-hero h1 { margin: 0; font-size: 1.7rem; letter-spacing: -0.01em; }
.vx-hero p { margin: .25rem 0 0; opacity: .7; }

.vx-stepper { display: flex; gap: .5rem; margin: 1.1rem 0 1.3rem; }
.vx-step { flex: 1; border: 1px solid rgba(128,128,128,.28); border-radius: 12px;
  padding: .7rem .85rem; display: flex; gap: .65rem; align-items: center;
  background: rgba(128,128,128,.05); }
.vx-dot { width: 26px; height: 26px; border-radius: 50%; flex: none; display: grid;
  place-items: center; font-size: .8rem; font-weight: 700;
  background: rgba(128,128,128,.25); color: inherit; }
.vx-step b { display: block; font-size: .92rem; }
.vx-step small { opacity: .65; line-height: 1.25; display: block; }
.vx-step.done { border-color: rgba(26,158,92,.55); background: rgba(26,158,92,.09); }
.vx-step.done .vx-dot { background: #1a9e5c; color: #fff; }
.vx-step.active { border-color: rgba(99,102,241,.8); background: rgba(99,102,241,.10);
  box-shadow: 0 0 0 3px rgba(99,102,241,.14); }
.vx-step.active .vx-dot { background: #6366f1; color: #fff; }
.vx-step.error { border-color: rgba(220,53,69,.7); background: rgba(220,53,69,.09); }
.vx-step.error .vx-dot { background: #dc3545; color: #fff; }

.vx-badge { display: inline-block; padding: .18rem .65rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; letter-spacing: .02em;
  background: rgba(128,128,128,.22); }
.vx-badge.language_selection { background: rgba(99,102,241,.2); color: #6366f1; }
.vx-badge.welcome { background: rgba(245,158,11,.2); color: #d98a06; }
.vx-badge.active { background: rgba(26,158,92,.2); color: #1a9e5c; }
.vx-badge.thinking { background: rgba(14,165,233,.2); color: #0e8fc9; }
.vx-badge.closed { background: rgba(220,53,69,.18); color: #dc3545; }

.vx-card { border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
  padding: 1rem 1.1rem; background: rgba(128,128,128,.04); }
.vx-card h4 { margin: 0 0 .5rem; font-size: .8rem; text-transform: uppercase;
  letter-spacing: .07em; opacity: .65; font-weight: 600; }

.vx-meta { display: grid; grid-template-columns: auto 1fr; gap: .25rem .9rem;
  font-size: .88rem; }
.vx-meta span:nth-child(odd) { opacity: .6; }
.vx-pill { display: inline-block; padding: .05rem .5rem; border-radius: 6px;
  background: rgba(128,128,128,.18); font-family: ui-monospace, monospace;
  font-size: .78rem; }

.vx-sys { font-size: .82rem; opacity: .85; padding: .15rem 0 .15rem .2rem;
  border-left: 3px solid rgba(128,128,128,.35); margin: .25rem 0 .25rem .2rem;
  padding-left: .7rem; }
.vx-sys.ok { border-color: #1a9e5c; }
.vx-sys.warn { border-color: #f59e0b; }
.vx-sys.err { border-color: #dc3545; }
.vx-t { opacity: .5; font-family: ui-monospace, monospace; font-size: .75rem;
  margin-right: .5rem; }
</style>
"""


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def esc(value: object) -> str:
    return html.escape(str(value))


def render_stepper(states: dict[str, str]) -> None:
    cells = []
    for number, (key, title, hint) in enumerate(STEPS, start=1):
        status = states[key]
        mark = "✓" if status == "done" else ("!" if status == "error" else str(number))
        cells.append(
            f'<div class="vx-step {status}"><div class="vx-dot">{mark}</div>'
            f"<div><b>{esc(title)}</b><small>{esc(hint)}</small></div></div>"
        )
    st.markdown(f'<div class="vx-stepper">{"".join(cells)}</div>', unsafe_allow_html=True)


def state_badge(state: str) -> str:
    label = state.replace("_", " ")
    return f'<span class="vx-badge {esc(state)}">{esc(label)}</span>'


def stamp(seconds: float) -> str:
    return f'<span class="vx-t">{seconds:5.1f}s</span>'


def wav_seconds(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, EOFError):
        return 0.0


def prompt_title(item: AudioItem) -> str:
    key = item.header.get("prompt")
    title = PROMPT_TITLES.get(key, "Assistant audio")
    language = LANGUAGE_NAMES.get(item.header.get("language") or "", "")
    return f"{title} ({language})" if language else title


def system_line(entry: Entry) -> tuple[str, str] | None:
    """(css class, html) for events shown as thin status lines."""

    kind, payload = entry.type, entry.payload

    if entry.source == "client":
        if kind == "select_language":
            name = LANGUAGE_NAMES.get(payload.get("language"), payload.get("language"))
            return "", f"{stamp(entry.at)}Caller tapped <b>{esc(name)}</b>"
        if kind == "reset":
            return (
                "warn",
                f"{stamp(entry.at)}Caller asked to choose again (<code>reset</code>)",
            )
        if kind == "audio":
            seconds = payload.get("seconds")
            length = f"{seconds:.1f}s, " if seconds else ""
            return "", (
                f"{stamp(entry.at)}Caller spoke ({length}"
                f"{payload.get('bytes', 0) / 1024:.0f} KB WAV)"
            )
        if kind == "text":
            return (
                "",
                f"{stamp(entry.at)}Caller said <b>“{esc(payload.get('text', ''))}”</b>",
            )
        return None

    if kind == "session.created":
        return "ok", (
            f"{stamp(entry.at)}Session <span class='vx-pill'>"
            f"{esc(str(payload.get('session_id', ''))[:8])}</span> created · protocol v"
            f"{esc(payload.get('protocol_version', '?'))}"
        )
    if kind == "state.changed":
        return "", f"{stamp(entry.at)}State → {state_badge(payload.get('state', '?'))}"
    if kind == "transcript":
        probability = payload.get("language_probability") or 0
        return "", (
            f"{stamp(entry.at)}Heard <b>“{esc(payload.get('text', ''))}”</b> · Whisper "
            f"thinks <span class='vx-pill'>{esc(payload.get('detected_language') or '?')}"
            f" {probability:.0%}</span>"
        )
    if kind == "language.rejected":
        return "warn", (
            f"{stamp(entry.at)}Language not recognised "
            f"(<span class='vx-pill'>{esc(payload.get('reason'))}</span>) · attempt "
            f"{payload.get('attempt')}, {payload.get('attempts_remaining')} left"
        )
    if kind == "language.selected":
        confidence = payload.get("confidence")
        pct = f" · {confidence:.0%}" if isinstance(confidence, int | float) else ""
        return "ok", (
            f"{stamp(entry.at)}Language set to <b>{esc(payload.get('display_name'))}</b>"
            f" via <span class='vx-pill'>{esc(payload.get('method', '?'))}</span>"
            f"{pct}"
            + (
                f" · <span class='vx-pill'>{esc(payload['reason'])}</span>"
                if payload.get("reason")
                else ""
            )
        )
    if kind == "language.failed":
        return "err", f"{stamp(entry.at)}{esc(payload.get('message'))}"
    if kind == "agent.thinking":
        return "", f"{stamp(entry.at)}Agent is thinking…"
    if kind == "error":
        severity = "err" if payload.get("fatal") else "warn"
        return severity, (
            f"{stamp(entry.at)}<b>{esc(payload.get('code'))}</b> — "
            f"{esc(payload.get('message'))}"
            + (" · call will close" if payload.get("fatal") else "")
        )
    if kind == "session.closed":
        return "", f"{stamp(entry.at)}Session closed ({esc(payload.get('reason'))})"
    return None


def render_timeline(snapshot: Snapshot) -> None:
    if not snapshot.entries:
        st.caption("Waiting for the first event…")
        return

    for entry in snapshot.entries:
        if entry.audio is not None:
            with st.chat_message("assistant"):
                item = entry.audio
                st.markdown(
                    f"{stamp(entry.at)}**{prompt_title(item)}** · "
                    f"{item.duration or wav_seconds(item.data):.1f}s · "
                    f"{len(item.data) / 1024:.0f} KB",
                    unsafe_allow_html=True,
                )
                st.audio(item.data, format="audio/wav")
            continue

        if entry.type == "agent.message":
            with st.chat_message("assistant"):
                st.markdown(
                    f"{stamp(entry.at)}{esc(entry.payload.get('text', ''))}",
                    unsafe_allow_html=True,
                )
                if entry.payload.get("metadata", {}).get("placeholder"):
                    st.caption(
                        "Placeholder echo agent — the hybrid RAG agent plugs in here."
                    )
            continue

        line = system_line(entry)
        if line is not None:
            css, body = line
            st.markdown(f'<div class="vx-sys {css}">{body}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Backend health
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3, show_spinner=False)
def fetch_backend(base: str) -> dict:
    out: dict = {"reachable": False}
    try:
        with httpx.Client(timeout=2.0) as client:
            ready = client.get(f"{base}/health/ready")
            out["reachable"] = True
            out["ready"] = ready.json()
            out["ready_ok"] = ready.status_code == 200
            status = client.get(f"{base}/api/v1/voice/status")
            if status.status_code == 200:
                out["status"] = status.json()
    except (httpx.HTTPError, ValueError) as exc:
        out["error"] = str(exc)
    return out


def render_backend_panel(base: str) -> dict:
    info = fetch_backend(base)

    if not info["reachable"]:
        st.error("Backend unreachable")
        st.caption(
            "Start it with `uvicorn app.main:app --port 8000`"
            + (f"\n\n{info.get('error')}" if info.get("error") else "")
        )
        return info

    components = info.get("ready", {}).get("components", {})
    if info.get("ready_ok"):
        st.success("Backend ready")
    else:
        st.warning("Backend starting")

    for name, detail in components.items():
        ok = bool(detail.get("ready"))
        extra = detail.get("name") or ""
        if name == "tts" and not ok:
            icon, extra = "⚪", "off · text replies only"  # intentional, not a fault
        else:
            icon = "🟢" if ok else "🔴"
        st.markdown(
            f"{icon} **{name}** <small>{esc(extra)}</small>", unsafe_allow_html=True
        )

    limits = (info.get("status") or {}).get("limits")
    if limits:
        with st.expander("Server limits", expanded=False):
            st.json(limits)
    return info


# --------------------------------------------------------------------------- #
# Caller controls
# --------------------------------------------------------------------------- #
def caller_controls(client: VoiceSessionClient, snapshot: Snapshot, view: dict) -> None:
    state = snapshot.state

    if snapshot.closed:
        st.info("The call has ended. Start a new one from the sidebar.")
        return

    if state in ("connecting", "welcome"):
        st.caption(
            "The welcome message is playing. The caller can answer as soon as it ends."
        )
        return

    if state == "language_selection":
        speak, typed, tap = st.tabs(["🎙️ Speak", "⌨️ Type what they say", "👆 Tap"])

        with speak:
            nonce = st.session_state.setdefault("mic_nonce", 0)
            clip = st.audio_input(
                "Say “Sinhala” or “English”", key=f"mic_{nonce}", sample_rate=16000
            )
            if clip is not None:
                payload = clip.getvalue()
                digest = hashlib.sha1(payload).hexdigest()
                if st.session_state.get("last_clip") != digest:
                    st.session_state["last_clip"] = digest
                    client.send_audio(payload, seconds=wav_seconds(payload))
                    st.session_state["mic_nonce"] = nonce + 1
                    st.rerun(scope="fragment")
            st.caption(
                "Speech goes through the same Whisper → detector path as a real call."
            )

        with typed:
            st.caption(
                "Skips the microphone and feeds a transcript to the language detector. "
                "Handy for testing odd Whisper outputs."
            )
            columns = st.columns(len(EXAMPLE_UTTERANCES))
            for column, (label, text) in zip(columns, EXAMPLE_UTTERANCES, strict=True):
                if column.button(label, key=f"ex_{text}", width="stretch"):
                    client.say(text)
            with st.form("say_form", clear_on_submit=True, border=False):
                text = st.text_input(
                    "Transcript", placeholder="e.g. “I'd like English please”"
                )
                if st.form_submit_button("Send", type="primary") and text.strip():
                    client.say(text.strip())

        with tap:
            st.caption("A UI button instead of speech: sends `select_language` directly.")
            left, right = st.columns(2)
            if left.button(
                "සිංහල · Sinhala", key="tap_si", width="stretch", type="primary"
            ):
                client.select_language("si-LK")
            if right.button("English", key="tap_en", width="stretch", type="primary"):
                client.select_language("en-US")
        return

    # ACTIVE / THINKING: the hand-off point to the hybrid RAG agent.
    st.markdown("**Hand-off preview**")
    st.caption("Everything after this point belongs to your hybrid RAG agent.")
    with st.form("ask_form", clear_on_submit=True, border=False):
        question = st.text_input(
            "Ask a question", placeholder="When does registration close?"
        )
        if st.form_submit_button("Ask", type="primary") and question.strip():
            client.say(question.strip())
    if st.button("↺ Choose language again", key="reset_btn"):
        client.reset()


def render_decision_card(snapshot: Snapshot, view: dict, limits: dict | None) -> None:
    rows: list[tuple[str, str]] = [("State", state_badge(snapshot.state))]

    max_retries = (limits or {}).get("language_max_retries")
    if view["language"]:
        rows.append(
            (
                "Language",
                f"<b>{esc(LANGUAGE_NAMES.get(view['language'], view['language']))}</b>"
                f" <span class='vx-pill'>{esc(view['language'])}</span>",
            )
        )
        rows.append(("Chosen by", f"<span class='vx-pill'>{esc(view['method'])}</span>"))
        if isinstance(view["confidence"], int | float):
            rows.append(("Confidence", f"{view['confidence']:.0%}"))
        if view["reason"]:
            rows.append(("Signal", f"<span class='vx-pill'>{esc(view['reason'])}</span>"))
        if view["selection_seconds"] is not None:
            rows.append(("Time to choose", f"{view['selection_seconds']:.1f}s"))
    else:
        rows.append(("Language", "—"))

    if view["attempts_remaining"] is not None:
        left = view["attempts_remaining"]
        attempts = f"{view['rejected']} · {left} attempt{'' if left == 1 else 's'} left"
    elif max_retries and not view["language"]:
        attempts = f"0 · {max_retries} attempts allowed"
    else:
        attempts = str(view["rejected"])
    rows.append(("Not recognised", attempts))

    if snapshot.session_id:
        rows.append(
            ("Session", f"<span class='vx-pill'>{esc(snapshot.session_id[:13])}</span>")
        )
    if snapshot.closed:
        rows.append(
            ("Closed", f"code {snapshot.close_code} {esc(snapshot.close_reason)}")
        )

    body = "".join(f"<span>{k}</span><span>{v}</span>" for k, v in rows)
    st.markdown(
        '<div class="vx-card"><h4>Call status</h4>'
        f'<div class="vx-meta">{body}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# The live call view
# --------------------------------------------------------------------------- #
def call_view(limits: dict | None) -> None:
    client: VoiceSessionClient | None = st.session_state.get("client")

    if client is None:
        st.markdown(
            '<div class="vx-card"><h4>No call in progress</h4>'
            "Press <b>Start call</b> in the sidebar to connect as a caller.</div>",
            unsafe_allow_html=True,
        )
        render_stepper({key: "pending" for key, *_ in STEPS})
        return

    snapshot = client.snapshot()
    view = analyse(snapshot)

    if snapshot.connect_error:
        st.error(snapshot.connect_error)
        if "403" in snapshot.connect_error:
            st.caption(
                "A 403 means the server's Origin check or per-IP limit refused "
                "this client. "
                "Check `SECURITY__*` settings or set the Origin in the sidebar."
            )

    render_stepper(step_states(snapshot, view))

    if view["failure"] and not snapshot.connect_error:
        st.error(view["failure"])

    controls, status = st.columns([1.35, 1], gap="large")
    with controls:
        st.markdown("##### Caller")
        caller_controls(client, snapshot, view)
    with status:
        render_decision_card(snapshot, view, limits)

    latest = snapshot.audio[-1] if snapshot.audio else None
    if latest is not None:
        st.markdown("##### Now playing")
        st.caption(f"{prompt_title(latest)} · autoplays when it arrives")
        # Only the newest clip autoplays; props stay stable between refreshes so
        # the browser does not restart it.
        st.audio(latest.data, format="audio/wav", autoplay=True)

    st.markdown("##### Call timeline")
    render_timeline(snapshot)

    with st.expander("Raw protocol events", expanded=False):
        rows = [
            {
                "t (s)": round(e.at, 2),
                "dir": "→ server" if e.source == "client" else "← client",
                "type": e.type,
                "payload": {k: v for k, v in e.payload.items() if k not in ("v", "ts")},
            }
            for e in snapshot.entries
        ]
        st.dataframe(rows, width="stretch", hide_index=True)

    if snapshot.closed and st.session_state.get("was_open"):
        st.session_state["was_open"] = False
        st.rerun()  # full rerun: lets the page drop the auto-refresh timer


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="Voice Assistant · Onboarding", page_icon="🎙️", layout="wide"
    )
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Connection")
        url = st.text_input("WebSocket URL", value=DEFAULT_URL, key="ws_url")
        origin = st.text_input(
            "Origin header (optional)",
            value="",
            help=(
                "Browsers send this automatically. Set it to test the "
                "server's origin allow-list."
            ),
        )
        base = http_base(url)
        backend = render_backend_panel(base)

        client = st.session_state.get("client")
        is_live = client is not None and client.is_open

        start, stop = st.columns(2)
        if start.button(
            "▶ Start call", type="primary", width="stretch", disabled=is_live
        ):
            if client is not None:
                client.close()
            new_client = VoiceSessionClient(url, origin=origin or None)
            new_client.start()
            st.session_state.update(client=new_client, was_open=True, last_clip=None)
            st.rerun()
        if stop.button("■ Hang up", width="stretch", disabled=not is_live):
            client.close()
            st.session_state["was_open"] = False
            st.rerun()

        st.divider()
        st.caption(
            "This page is a plain WebSocket client. It does not import any server code, "
            "so it shows exactly what the Angular app will receive."
        )

    st.markdown(
        '<div class="vx-hero"><h1>🎙️ Voice assistant · first minute of a call</h1>'
        "<p>Welcome message → language selection → confirmation. "
        "Everything after this belongs to the hybrid RAG agent.</p></div>",
        unsafe_allow_html=True,
    )

    limits = (backend.get("status") or {}).get("limits")
    client = st.session_state.get("client")
    live = client is not None and not client.snapshot().closed

    runner = (
        st.fragment(call_view, run_every=REFRESH_SECONDS)
        if live
        else st.fragment(call_view)
    )
    runner(limits)


main()
