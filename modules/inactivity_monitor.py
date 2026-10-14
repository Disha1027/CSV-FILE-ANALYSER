"""Client-side inactivity detection for the Streamlit app."""

import streamlit as st


_IDLE_COMPONENT = st.components.v2.component(
    "csv_file_analyser_idle_timeout",
    html='<div aria-live="polite" style="display:none"></div>',
    js="""
export default function(component) {
    const { data, setTriggerValue } = component;
    const timeoutSeconds = Math.max(1, Number(data?.timeout_seconds ?? 20));
    let timeoutHandle;
    let expired = false;
    let lastPointerMove = 0;

    const expire = () => {
        if (expired) return;
        expired = true;
        setTriggerValue("inactivity_timeout", true);
    };

    const resetTimer = (event) => {
        if (event.type === "pointermove") {
            const now = Date.now();
            if (now - lastPointerMove < 250) return;
            lastPointerMove = now;
        }
        if (expired) return;
        window.clearTimeout(timeoutHandle);
        timeoutHandle = window.setTimeout(expire, timeoutSeconds * 1000);
    };

    const events = ["pointerdown", "pointermove", "keydown", "wheel", "touchstart", "input", "change"];
    events.forEach((name) => document.addEventListener(name, resetTimer, true));
    resetTimer({ type: "initial" });

    return () => {
        window.clearTimeout(timeoutHandle);
        events.forEach((name) => document.removeEventListener(name, resetTimer, true));
    };
}
""",
)


def inactivity_monitor(timeout_seconds: int = 20, key: str = "session_idle_monitor"):
    """Return the inactivity timeout event, if the browser has been idle."""
    result = _IDLE_COMPONENT(
        data={"timeout_seconds": timeout_seconds},
        key=key,
        on_inactivity_timeout_change=lambda: None,
    )
    return bool(getattr(result, "inactivity_timeout", False))