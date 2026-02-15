import os

import streamlit as st
import google.generativeai as genai


st.set_page_config(page_title="Gemini Smoke Test", page_icon=":test_tube:")
st.title("Gemini API Smoke Test")


def get_gemini_key() -> str:
    """Read key from Streamlit secrets first, then env."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY", "")


api_key = get_gemini_key()
if not api_key:
    st.error("No GEMINI_API_KEY found in Streamlit secrets or environment.")
    st.stop()

genai.configure(api_key=api_key)

if st.button("Run Gemini checks", type="primary"):
    with st.spinner("Checking Gemini API..."):
        try:
            models = list(genai.list_models())
            st.success("API key is valid: list_models() worked.")

            st.subheader("Available models")
            for model in models:
                methods = ", ".join(getattr(model, "supported_generation_methods", []) or [])
                st.write(f"- `{model.name}` | methods: {methods}")

            # Prefer your target model if available; otherwise fallback.
            preferred_model = "gemini-2.5-flash"
            model_names = [m.name.replace("models/", "") for m in models]
            chosen_model = preferred_model if preferred_model in model_names else None

            if chosen_model is None:
                # Pick first model that can generate content
                for model in models:
                    methods = getattr(model, "supported_generation_methods", []) or []
                    if "generateContent" in methods:
                        chosen_model = model.name.replace("models/", "")
                        break

            if chosen_model is None:
                st.warning("Could not find a model supporting generateContent.")
            else:
                st.info(f"Testing prompt generation with `{chosen_model}`...")
                llm = genai.GenerativeModel(chosen_model)
                reply = llm.generate_content("Reply with exactly: Gemini API OK")
                text = getattr(reply, "text", None)
                if text:
                    st.success("Prompt generation worked.")
                    st.write("Model reply:", text)
                else:
                    st.warning("Generation returned no text, but request succeeded.")

        except Exception as exc:
            st.error("Gemini API check failed.")
            st.exception(exc)
