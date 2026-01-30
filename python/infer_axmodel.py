from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from typing import List, Tuple
from axengine import InferenceSession
from ml_dtypes import bfloat16
from utils.infer_func import InferManager
import argparse
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


if __name__ == "__main__":

    """
    python3 infer_axmodel.py  --vit_model vit-models/internvl_vit_model_1x3x448x448.axmodel --images examples/image_0.jpg
    """
    prompt = None
    parser = argparse.ArgumentParser(description="Model configuration parameters")
    parser.add_argument("--hf_model", type=str, default="./HY-MT1.5-1.8B",
                        help="Path to HuggingFace model")
    parser.add_argument("--axmodel_path", type=str, default="./HY-MT1.5-1.8B_GPTQ_INT4_ACC_axmodel",
                        help="Path to save compiled axmodel of llama model")
    parser.add_argument("-q", "--question", type=str, default="It’s on the house.",
                        help="Your question that you want to ask the model.")
    args = parser.parse_args()

    hf_model_path = args.hf_model
    axmodel_path = args.axmodel_path
    prompt = args.question

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeds = np.load(os.path.join(axmodel_path, "model.embed_tokens.weight.npy"))

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    # model = AutoModelForCausalLM.from_pretrained(
    #     hf_model_path,
    # ).to(device)

    messages = [
        {"role": "user", "content": f"Translate the following segment into Chinese, without additional explanation.\n\n{prompt}"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    )
    # import pdb; pdb.set_trace()
    # input_ids = tokenized_chat.input_ids
    token_ids = input_ids[0].cpu().numpy().tolist()
    token_len = len(token_ids)
    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = prefill_data.astype(bfloat16)

    cfg = config

    eos_token_id = None
    if isinstance(cfg.eos_token_id, list) and len(cfg.eos_token_id) > 1:
        eos_token_id = cfg.eos_token_id

    slice_len = 128
    prefill_max_len = 1024 - 1
    max_seq_len = 2048 - 1  # prefill + decode max length

    imer = InferManager(cfg, axmodel_path, max_seq_len=max_seq_len) # prefill + decode max length
    # import pdb; pdb.set_trace()
    token_ids = imer.prefill(tokenizer, token_ids, prefill_data, slice_len=slice_len)
    imer.decode(tokenizer, token_ids, embeds, slice_len=slice_len, eos_token_id=eos_token_id)
    print("\n")