import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_quantization_config(QUANTIZATION_BITS):
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None

    if QUANTIZATION_BITS == 4:
        # 4-bit quantization (most memory efficient)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4",  # NormalFloat4 - better for LLMs
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        # 8-bit quantization (balanced)
        config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        raise ValueError(
            f"Invalid QUANTIZATION_BITS: {QUANTIZATION_BITS}. Use 4, 8, or None"
        )

    return config


def load_model_and_tokenizer(model_name, device, QUANTIZATION_BITS):
    """Load model with optional quantization"""
    print(f"\nLoading model {model_name}...")
    print(f"Device: {device}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")

        # Get quantization config
        quant_config = get_quantization_config(QUANTIZATION_BITS)

        # Load model
        print("Loading model (this may take 2-3 minutes)...")

        if quant_config is not None:
            # Quantized model loading (only works with CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            # Non-quantized model loading
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=torch.float16, low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:  # CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, dtype=torch.float32, low_cpu_mem_usage=True
                )
                model = model.to(device)

        model.eval()

        # Print model info
        print("✓ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(
                f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved"
            )

            # Check if using quantization
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print("  Running on Apple Metal (MPS)")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print(
            "2. Model license not accepted - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct"
        )
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise


def get_model_prediction(model, tokenizer, prompt, MAX_NEW_TOKENS):
    """Get model's prediction for multiple-choice question with timing"""
    cpu_start = time.process_time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    cpu_time_seconds = time.process_time() - cpu_start

    gpu_time_seconds = 0
    gpu_start_event = None
    gpu_end_event = None
    if torch.cuda.is_available() and model.device.type == "cuda":
        torch.cuda.synchronize()
        gpu_start_event = torch.cuda.Event(enable_timing=True)
        gpu_end_event = torch.cuda.Event(enable_timing=True)
        gpu_start_event.record()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
        )

    if gpu_start_event and gpu_end_event:
        gpu_end_event.record()
        torch.cuda.synchronize()
        gpu_time_seconds = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0

    cpu_decode_start = time.process_time()
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    cpu_time_seconds += time.process_time() - cpu_decode_start

    answer = generated_text.strip()[:1].upper()

    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break

    return answer, cpu_time_seconds, gpu_time_seconds
