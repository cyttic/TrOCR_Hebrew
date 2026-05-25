from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoImageProcessor,
    AutoTokenizer,
    BertLMHeadModel,
)

OUTPUT_DIR = "trocr-hebrew-untrained"

print("Loading TrOCR (extracting ViT encoder)...")
trocr = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-handwritten", use_safetensors=True
)
encoder = trocr.encoder

print("Loading Hebrew decoder (DictaBERT)...")
decoder = BertLMHeadModel.from_pretrained(
    "dicta-il/dictabert",
    is_decoder=True,
    add_cross_attention=True,
    use_safetensors=True,
)

print("Combining encoder + decoder...")
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

# configure special tokens so generation works correctly
tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert")
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# recommended generation settings (must live in generation_config, not model.config)
model.generation_config.max_new_tokens = 64
model.generation_config.early_stopping = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 4

print("Saving model...")
model.save_pretrained(OUTPUT_DIR)

print("Building processor (TrOCR image processor + Hebrew tokenizer)...")
image_processor = AutoImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nDone. Model saved to: {OUTPUT_DIR}/")
print("You can now upload it to HuggingFace with:")
print(f"  huggingface-cli upload <your-username>/trocr-hebrew-untrained {OUTPUT_DIR}/")
print("\nNote: decoder is DictaBERT (dicta-il/dictabert) — Hebrew BERT with safetensors support.")
