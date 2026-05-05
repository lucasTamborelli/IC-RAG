import tiktoken
import streamlit as sl

def render_tab(type, answer, tokens):
        sl.markdown("**Resposta:**")
        sl.write(answer)
        sl.text(f"Tokens: {tokens}")

def n_tokens(input: str, output: str, model_name: str):
        enc = tiktoken.get_encoding(model_name)
        tokens_in = enc.encode(input)
        tokens_out = enc.encode(output)
        return [len(tokens_in), len(tokens_out)]
