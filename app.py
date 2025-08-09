import streamlit as st
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="App-maker", page_icon="✨", layout="centered")

st.title("✨ App-maker — Lumi Enjoy Mode")
st.caption("Deploy-ready Streamlit template on Render")

st.markdown("**Try it:** upload an image and add some context, then click Generate.")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp"])
context = st.text_input("Context (optional)", placeholder="e.g., BlackPink vibe, product, mood")

def simple_caption(ctx: str) -> str:
    base = "A clean, modern caption"
    if ctx:
        return f"{base} with {ctx.strip()}."
    return base + "."

col1, col2 = st.columns(2)
with col1:
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Preview", use_column_width=True)
with col2:
    if st.button("Generate"):
        st.success(simple_caption(context))

st.divider()
st.write("✅ Ready for Render: binds to 0.0.0.0 and $PORT via startCommand in `.render.yaml`.")
