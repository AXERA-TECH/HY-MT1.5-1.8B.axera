import argparse
import os
import socket
import numpy as np
import gradio as gr
from transformers import AutoConfig, AutoTokenizer
from ml_dtypes import bfloat16

from utils.infer_func import InferManager

DEFAULT_LANGUAGES = [
    "English",
    "Chinese",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Italian",
    "Portuguese",
    "Russian",
    "Arabic",
    "Hindi",
    "Bengali",
    "Thai",
    "Vietnamese",
    "Indonesian",
    "Turkish",
    "Polish",
    "Dutch",
    "Swedish",
    "Danish",
    "Norwegian",
    "Finnish",
    "Greek",
    "Czech",
    "Hungarian",
    "Romanian",
    "Ukrainian",
    "Malay",
    "Filipino",
    "Urdu",
    "Hebrew",
    "Persian",
]


def build_prompt(source_text: str, target_language: str, use_zh_template: bool) -> str:
    if use_zh_template:
        return (
            f"将以下文本翻译为{target_language}，注意只需要输出翻译后的结果，不要额外解释：\n"
            f"{source_text}"
        )
    return (
        f"Translate the following segment into {target_language}, without additional explanation.\n"
        f"{source_text}"
    )


def _get_ipv4_address() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def create_demo(hf_model: str, axmodel_path: str):
    embeds_path = os.path.join(axmodel_path, "model.embed_tokens.weight.npy")
    if not os.path.exists(embeds_path):
        raise FileNotFoundError(f"Missing embeddings file: {embeds_path}")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    config = AutoConfig.from_pretrained(hf_model, trust_remote_code=True)
    embeds = np.load(embeds_path)

    eos_token_id = None
    if isinstance(config.eos_token_id, list) and len(config.eos_token_id) > 1:
        eos_token_id = config.eos_token_id

    max_seq_len = 2048 - 1
    imer = InferManager(config, axmodel_path, max_seq_len=max_seq_len)

    def translate(
        text,
        target_language,
        use_zh_template,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
    ):
        if not text or not text.strip():
            return ""

        prompt = build_prompt(text.strip(), target_language, use_zh_template)
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        token_ids = input_ids[0].cpu().numpy().tolist()
        prefill_data = np.take(embeds, token_ids, axis=0).astype(bfloat16)

        slice_len = 128
        token_ids = imer.prefill(
            tokenizer,
            token_ids,
            prefill_data,
            slice_len=slice_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        output = imer.decode(
            tokenizer,
            token_ids,
            embeds,
            slice_len=slice_len,
            eos_token_id=eos_token_id,
            stream=False,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        )
        return output.strip()

    def translate_stream(
        text,
        target_language,
        use_zh_template,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        max_new_tokens,
    ):
        if not text or not text.strip():
            yield ""
            return

        prompt = build_prompt(text.strip(), target_language, use_zh_template)
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        token_ids = input_ids[0].cpu().numpy().tolist()
        prefill_data = np.take(embeds, token_ids, axis=0).astype(bfloat16)

        slice_len = 128
        token_ids = imer.prefill(
            tokenizer,
            token_ids,
            prefill_data,
            slice_len=slice_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        for text_so_far in imer.decode_stream(
            tokenizer,
            token_ids,
            embeds,
            slice_len=slice_len,
            eos_token_id=eos_token_id,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
        ):
            yield text_so_far.strip()

    with gr.Blocks(title="HY-MT1.5-1.8B Multilingual Translation") as demo:
        gr.Markdown("## HY-MT1.5-1.8B Multilingual Translation")

        with gr.Group():
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Please enter the text you want to translate...",
                lines=6,
            )

        with gr.Group():
            with gr.Row(equal_height=True):
                target_language = gr.Dropdown(
                    choices=DEFAULT_LANGUAGES,
                    value="English",
                    label="Target Language",
                )
                use_zh_template = gr.Checkbox(
                    label="Use Chinese Prompt Template",
                    value=False,
                )
        with gr.Group():
            with gr.Row(equal_height=True):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.05,
                    label="Top-p",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    label="Top-k",
                )

        with gr.Group():
            with gr.Row(equal_height=True):
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=1.5,
                    value=1.05,
                    step=0.01,
                    label="Repetition Penalty",
                )
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=1024,
                    value=512,
                    step=1,
                    label="Max New Tokens",
                )

        translate_btn = gr.Button("Translate", variant="primary")
        output_text = gr.Textbox(
            label="Translation Result",
            lines=6,
            interactive=False,
        )

        translate_btn.click(
            translate_stream,
            inputs=[
                input_text,
                target_language,
                use_zh_template,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                max_new_tokens,
            ],
            outputs=output_text,
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="HY-MT1.5-1.8B Gradio Demo")
    parser.add_argument(
        "--hf_model",
        type=str,
        default="./HY-MT1.5-1.8B",
        help="Path to HuggingFace model",
    )
    parser.add_argument(
        "--axmodel_path",
        type=str,
        default="./HY-MT1.5-1.8B_GPTQ_INT4_ACC_axmodel",
        help="Path to compiled axmodel directory",
    )
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=7860)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_demo(args.hf_model, args.axmodel_path)
    ipv4 = _get_ipv4_address()
    print(f"* Running on local URL:  http://{ipv4}:{args.server_port}")
    app.launch(server_name=args.server_name, server_port=args.server_port)
