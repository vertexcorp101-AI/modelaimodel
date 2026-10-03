"""Fine-tune SmolLM2-360M-Instruct into tiny-Vertex's voice (LoRA).

Run on Colab (free T4, ~20-40 min) or overnight on CPU.
Colab cells are in README_COLAB.md. Output: ./tinyvertex-360m-merged
(a full fp16 model dir), ready for GGUF conversion (see README_COLAB.md).

Requires: transformers, peft, trl, accelerate, datasets.
  pip install -U transformers peft trl accelerate datasets
"""
import os

MODEL_ID = os.environ.get("SFT_BASE", "HuggingFaceTB/SmolLM2-360M-Instruct")
DATA = os.environ.get("SFT_DATA", "tiny_vertex_dialogues.jsonl")
OUT = os.environ.get("SFT_OUT", "tinyvertex-360m-merged")

EPOCHS = float(os.environ.get("SFT_EPOCHS", "2"))
LR = float(os.environ.get("SFT_LR", "1e-4"))
MAX_LEN = int(os.environ.get("SFT_MAXLEN", "512"))


def main():
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    try:
        # trl >= 0.9: SFTConfig + processing_class (needs pip install -U trl)
        from trl import SFTTrainer, SFTConfig
        _TRL_NEW = True
    except ImportError:
        # older trl: plain TrainingArguments, tokenizer= kwarg
        from trl import SFTTrainer
        from transformers import TrainingArguments as SFTConfig
        _TRL_NEW = False

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"device: {'cuda' if use_cuda else 'cpu'}, dtype: {dtype}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype,
        device_map="auto" if use_cuda else None,
    )
    if not use_cuda:
        model.gradient_checkpointing_enable()
    else:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=DATA)["train"]
    print(f"examples: {len(ds)}")

    def fmt(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}

    ds = ds.map(fmt, remove_columns=ds.column_names)

    base_kwargs = dict(
        output_dir="sft-out",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2 if use_cuda else 1,
        gradient_accumulation_steps=4 if use_cuda else 8,
        learning_rate=LR,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        fp16=use_cuda,
    )
    if _TRL_NEW:
        args = SFTConfig(**base_kwargs, max_length=MAX_LEN,
                         packing=False, dataset_text_field="text")
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                             processing_class=tok)
    else:
        args = SFTConfig(**base_kwargs)
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                             tokenizer=tok, dataset_text_field="text",
                             max_seq_length=MAX_LEN, packing=False)
    trainer.train()

    merged = model.merge_and_unload()
    merged.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"saved merged model to {OUT}")


if __name__ == "__main__":
    main()
