"""Streamlit chat UI for ragu: chat in the main area, retrieved recipes in the sidebar."""

import json
import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("RAGU_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ragù", page_icon="🍝", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content}
if "recipes" not in st.session_state:
    st.session_state.recipes = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex  # one conversation per browser session
if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = set()  # trace_ids already rated, so we hide the widget


def recipe_card(r):
    with st.container(border=True):
        if r.get("image"):
            st.image(r["image"], use_container_width=True)
        st.markdown(f"**{r['name']}**")

        bits = []
        if r.get("rating") is not None:
            bits.append(f"⭐ {r['rating']:.1f} ({r['n_ratings']})")
        if r.get("total_time") is not None:
            bits.append(f"⏱ {r['total_time']} min")
        if bits:
            st.caption("  ·  ".join(bits))

        nutrition = []
        if r.get("calories") is not None:
            nutrition.append(f"{round(r['calories'])} kcal")
        for key, label in (("protein", "P"), ("carbs", "C"), ("fat", "F")):
            if r.get(key) is not None:
                nutrition.append(f"{label}{round(r[key])}")
        if nutrition:
            st.caption(" · ".join(nutrition))

        tags = [t for t in [r.get("category"), *r.get("keywords", [])[:3]] if t]
        if tags:
            st.caption(" ".join(f"`{t}`" for t in tags))

        with st.expander("Ingredients & steps"):
            ingredients = [str(x) for x in r.get("ingredients", []) if x]
            if ingredients:
                st.markdown("**Ingredients**")
                st.markdown("\n".join(f"- {x}" for x in ingredients))
            steps = [str(s) for s in r.get("instructions", []) if s]
            if steps:
                st.markdown("**Steps**")
                st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))


def feedback_ui(idx, trace_id):
    """Thumbs + optional comment under an assistant turn, sent to /feedback as a Langfuse score."""
    if trace_id in st.session_state.feedback_sent:
        st.caption("✓ Thanks for the feedback")
        return
    # st.feedback("thumbs") returns 1 for 👍, 0 for 👎, None if untouched.
    score = st.feedback("thumbs", key=f"thumb_{idx}")
    comment = st.text_input(
        "Comment", key=f"cmt_{idx}",
        label_visibility="collapsed", placeholder="Add a comment (optional)",
    )
    if st.button("Send feedback", key=f"send_{idx}"):
        if score is None:
            st.warning("Pick 👍 or 👎 first.")
            return
        try:
            httpx.post(
                f"{API_URL}/feedback",
                json={"trace_id": trace_id, "value": bool(score), "comment": comment},
                timeout=10,
            ).raise_for_status()
        except httpx.HTTPError:
            st.error("Couldn't send feedback. Is the API running?")
            return
        st.session_state.feedback_sent.add(trace_id)
        st.rerun()


with st.sidebar:
    st.subheader("Recipes used")
    if not st.session_state.recipes:
        st.caption("Ask something to see the recipes behind the answer here.")
    for r in st.session_state.recipes:
        recipe_card(r)

st.title("🍝 Ragù")
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("trace_id"):
            feedback_ui(i, m["trace_id"])

if prompt := st.chat_input("e.g. high-protein dinner under 600 kcal, ready in 30 min"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        # One placeholder rewritten per status frame, so progress replaces itself in place.
        status = st.empty()
        result = None
        try:
            with httpx.stream(
                "POST",
                f"{API_URL}/chat",
                json={"question": prompt, "thread_id": st.session_state.thread_id},
                timeout=60,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line.removeprefix("data: "))
                    if event["type"] == "status":
                        status.markdown(f"*{event['text']}*")
                    else:
                        result = event
        except httpx.HTTPError:
            st.error(f"Can't reach the API at {API_URL}. Is it running? (uvicorn api:app)")
            st.stop()
        status.empty()
        st.markdown(result["answer"])
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "trace_id": result.get("trace_id")}
    )
    st.session_state.recipes = result["recipes"]
    st.rerun()
