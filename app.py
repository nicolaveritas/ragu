"""Streamlit chat UI for ragu: chat in the main area, retrieved recipes in the sidebar."""

import os

import httpx
import streamlit as st

API_URL = os.getenv("RAGU_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Ragù", page_icon="🍝", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content}
if "recipes" not in st.session_state:
    st.session_state.recipes = []
if "constraints" not in st.session_state:
    st.session_state.constraints = []
if "filter_relaxed" not in st.session_state:
    st.session_state.filter_relaxed = False


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


with st.sidebar:
    st.subheader("Retrieved recipes")
    if st.session_state.constraints:
        chips = " ".join(
            f"`{c['field']} {c['op']} {c['value']:g}`" for c in st.session_state.constraints
        )
        st.caption(f"Detected constraints: {chips}")
    if st.session_state.filter_relaxed:
        st.warning("No recipe matched all your limits — showing the closest matches.")
    if not st.session_state.recipes:
        st.caption("Ask something to see matching recipes here.")
    for r in st.session_state.recipes:
        recipe_card(r)

st.title("🍝 Ragù")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("e.g. high-protein dinner under 600 kcal, ready in 30 min"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Searching recipes…"):
        try:
            result = httpx.post(f"{API_URL}/chat", json={"question": prompt}, timeout=60).json()
        except httpx.HTTPError:
            st.error(f"Can't reach the API at {API_URL}. Is it running? (uvicorn api:app)")
            st.stop()
        st.markdown(result["answer"])
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.session_state.recipes = result["recipes"]
    st.session_state.constraints = result["constraints"]
    st.session_state.filter_relaxed = result["filter_relaxed"]
    st.rerun()
