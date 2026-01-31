"""
Llama 3.2-1B MMLU Evaluation Script (Laptop Optimized with Quantization)

This script evaluates Llama 3.2-1B on the MMLU benchmark.
Optimized for laptops with 4-bit or 8-bit quantization to reduce memory usage.

Quantization options:
- 4-bit: ~1.5 GB VRAM/RAM (default for laptop)
- 8-bit: ~2.5 GB VRAM/RAM
- No quantization: ~5 GB VRAM/RAM

Usage:
1. Install: pip install transformers torch datasets accelerate tqdm bitsandbytes matplotlib
2. Login: huggingface-cli login
3. Run: python llama_mmlu_eval.py

Set QUANTIZATION_BITS below to choose quantization level.

Environment Variables:
- MODEL_NAME: Comma-separated model names (default: meta-llama/Llama-3.2-1B-Instruct)
- QUICK_TEST: Run on first 2 subjects only (True/False, default: False)
- VERBOSE_OUTPUT: Save detailed Q&A to file (True/False, default: False)
- QUANTIZATION_BITS: 4, 8, or None (default: None)
- USE_GPU: Use GPU if available (True/False, default: True)

Command Line Options:
- --verbose: Save each question and answer to verbose output file
- --quick-test: Run on first 2 subjects only
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import argparse
import platform
import time
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
OUTPUT_DIRS = {
    "evals": os.path.join("Results", "evals"),      # JSON results
    "images": os.path.join("Results", "images"),    # Graph images
    "logs": os.path.join("Results", "logs")         # Verbose output
}

def setup_output_directories():
    """Create output directories if they don't exist"""
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Accept comma-separated models: MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct,meta-llama/Llama-2-7B"
MODEL_NAMES = [m.strip() for m in os.environ.get("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct").split(",")]

# GPU settings
# If True, will attempt to use the best available GPU (CUDA for NVIDIA, MPS for Apple Silicon)
# If False, will always use CPU regardless of available hardware
USE_GPU = os.environ.get("USE_GPU", True)  # Set to False to force CPU-only execution
if isinstance(USE_GPU, str):
    USE_GPU = eval(USE_GPU)

MAX_NEW_TOKENS = 1

# Quantization settings
# Options: 4, 8, or None (default is None for full precision)
#
# To enable quantization, change QUANTIZATION_BITS to one of the following:
#   QUANTIZATION_BITS = 4   # 4-bit quantization: ~1.5 GB memory (most memory efficient)
#   QUANTIZATION_BITS = 8   # 8-bit quantization: ~2.5 GB memory (balanced quality/memory)
#   QUANTIZATION_BITS = None  # No quantization: ~5 GB memory (full precision, best quality)
#
# Notes:
# - Quantization requires the 'bitsandbytes' package: pip install bitsandbytes
# - Quantization only works with CUDA (NVIDIA GPUs), not with Apple Metal (MPS)
# - If using Apple Silicon, quantization will be automatically disabled

QUANTIZATION_BITS = os.environ.get("QUANTIZATION_BITS", None) # Change to 4 or 8 to enable quantization
if QUANTIZATION_BITS is not None:
    QUANTIZATION_BITS = int(QUANTIZATION_BITS)

# Testing and output options
QUICK_TEST = os.environ.get("QUICK_TEST", "False")  # If True, run on first 2 subjects only
if isinstance(QUICK_TEST, str):
    QUICK_TEST = QUICK_TEST.lower() in ('true', '1', 'yes')

VERBOSE_OUTPUT = os.environ.get("VERBOSE_OUTPUT", "False")  # If True, save detailed Q&A to file
if isinstance(VERBOSE_OUTPUT, str):
    VERBOSE_OUTPUT = VERBOSE_OUTPUT.lower() in ('true', '1', 'yes')

# For quick testing, you can reduce this list
MMLU_SUBJECTS = [
    "astronomy", 
    "business_ethics", 
    "anatomy", 
    "computer_security",
    "high_school_biology", 
    "high_school_mathematics", 
    "philosophy",
    "us_foreign_policy", 
    "machine_learning", 
    "professional_law",
    # Additional subjects (commented out by default)
    # "abstract_algebra", "clinical_knowledge", "college_biology", "college_chemistry",
    # "college_computer_science", "college_mathematics", "college_medicine",
    # "college_physics", "conceptual_physics",
    # "econometrics", "electrical_engineering", "elementary_mathematics",
    # "formal_logic", "global_facts",
    # "high_school_chemistry", "high_school_computer_science",
    # "high_school_european_history", "high_school_geography",
    # "high_school_government_and_politics", "high_school_macroeconomics",
    # "high_school_microeconomics",
    # "high_school_physics", "high_school_psychology", "high_school_statistics",
    # "high_school_us_history", "high_school_world_history", "human_aging",
    # "human_sexuality", "international_law", "jurisprudence",
    # "logical_fallacies", "management", "marketing",
    # "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    # "nutrition", "prehistory", "professional_accounting",
    # "professional_medicine", "professional_psychology",
    # "public_relations", "security_studies", "sociology",
    # "virology", "world_religions"
]


def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""

    # If GPU is disabled, always use CPU
    if not USE_GPU:
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for Apple Silicon with Metal
    if torch.backends.mps.is_available():
        # Check if we're actually on Apple ARM
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"

        if is_apple_arm:
            # Metal is available but incompatible with quantization
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                print("Metal Performance Shaders (MPS) is incompatible with quantization.")
                print(f"You have USE_GPU = True and QUANTIZATION_BITS = {QUANTIZATION_BITS}")
                print("")
                print("Please choose one of the following options:")
                print("  1. Set USE_GPU = False to use CPU with quantization")
                print("  2. Set QUANTIZATION_BITS = None to use Metal without quantization")
                print("="*70 + "\n")
                sys.exit(1)
            return "mps"

    # Default to CPU
    return "cpu"




def check_environment():
    global QUANTIZATION_BITS
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    # Check if in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    # Check system info
    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")

    # Detect and set device
    device = detect_device()

    # Check device
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
        print("✓ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("⚠️  No GPU detected - running on CPU")
       
    # Check quantization support

    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"❌ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"❌ Apple METAL is incompatible with quantization")
            print("✓ Quantization disabled - loading full precision model")
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("✓ Quantization disabled - loading full precision model")
    
    # Check HF authentication
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authenticated")
        else:
            print("⚠️  No Hugging Face token found")
            print("Run: huggingface-cli login")
    except:
        print("⚠️  Could not check Hugging Face authentication")

    # Print environment variables
    print("\nEnvironment Variables:")
    print(f"  MODEL_NAMES: {', '.join(MODEL_NAMES)}")
    print(f"  USE_GPU: {USE_GPU}")
    print(f"  QUANTIZATION_BITS: {QUANTIZATION_BITS}")

    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Models to evaluate: {len(MODEL_NAMES)}")
    for i, model_name in enumerate(MODEL_NAMES, 1):
        print(f"  {i}. {model_name}")
    print(f"Device: {device}")
    if QUANTIZATION_BITS is not None:
        print(f"Quantization: {QUANTIZATION_BITS}-bit")
        if QUANTIZATION_BITS == 4:
            print(f"Expected memory per model: ~1.5 GB")
        elif QUANTIZATION_BITS == 8:
            print(f"Expected memory per model: ~2.5 GB")
    else:
        print(f"Quantization: None (full precision)")
        if device == "cuda":
            print(f"Expected memory per model: ~2.5 GB (FP16)")
        elif device == "mps":
            print(f"Expected memory per model: ~2.5 GB (FP16)")
        else:
            print(f"Expected memory per model: ~5 GB (FP32)")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")

    print("="*70 + "\n")
    return in_colab, device


def get_quantization_config():
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None
    
    if QUANTIZATION_BITS == 4:
        # 4-bit quantization (most memory efficient)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4"  # NormalFloat4 - better for LLMs
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        # 8-bit quantization (balanced)
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        raise ValueError(f"Invalid QUANTIZATION_BITS: {QUANTIZATION_BITS}. Use 4, 8, or None")
    
    return config


def load_model_and_tokenizer(model_name, device):
    """Load model with optional quantization"""
    print(f"\nLoading model {model_name}...")
    print(f"Device: {device}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")

        # Get quantization config
        quant_config = get_quantization_config()

        # Load model
        print("Loading model (this may take 2-3 minutes)...")

        if quant_config is not None:
            # Quantized model loading (only works with CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Non-quantized model loading
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:  # CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
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
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

            # Check if using quantization
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print(f"  Running on Apple Metal (MPS)")

        return model, tokenizer
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print("2. Model license not accepted - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(model, tokenizer, prompt):
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
            temperature=1.0
        )
    
    if gpu_start_event and gpu_end_event:
        gpu_end_event.record()
        torch.cuda.synchronize()
        gpu_time_seconds = gpu_start_event.elapsed_time(gpu_end_event) / 1000.0
    
    cpu_decode_start = time.process_time()
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    cpu_time_seconds += time.process_time() - cpu_decode_start
    
    answer = generated_text.strip()[:1].upper()
    
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
    
    return answer, cpu_time_seconds, gpu_time_seconds


def evaluate_subject(model, tokenizer, subject, verbose_output=False, verbose_file_handle=None):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    
    # Start wall clock timing
    wall_clock_start = datetime.now()
    
    cpu_time_seconds = 0
    gpu_time_seconds = 0
    question_results = {}
    
    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices)
        predicted_answer, cpu_time, gpu_time = get_model_prediction(model, tokenizer, prompt)
        cpu_time_seconds += cpu_time
        gpu_time_seconds += gpu_time
        
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct += 1
        total += 1
        
        # Generate hash for question to track across models
        # MD5 is fast and sufficient for this purpose
        question_hash = hashlib.md5(question.encode()).hexdigest()
        question_results[question_hash] = is_correct
        
        # Write verbose output if enabled
        if verbose_output and verbose_file_handle:
            verbose_file_handle.write(f"\n{'='*70}\n")
            verbose_file_handle.write(f"Subject: {subject} | Question {total}\n")
            verbose_file_handle.write(f"{'='*70}\n")
            verbose_file_handle.write(f"Q: {question}\n\n")
            for i, choice in enumerate(choices):
                verbose_file_handle.write(f"  {['A','B','C','D'][i]}. {choice}\n")
            verbose_file_handle.write(f"\nModel Answer: {predicted_answer}\n")
            verbose_file_handle.write(f"Correct Answer: {correct_answer}\n")
            verbose_file_handle.write(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}\n")
            verbose_file_handle.flush()
    
    # End wall clock timing
    wall_clock_end = datetime.now()
    wall_clock_seconds = (wall_clock_end - wall_clock_start).total_seconds()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    print(f"  Wall Clock: {wall_clock_seconds:.2f}s | CPU: {cpu_time_seconds:.2f}s | GPU: {gpu_time_seconds:.2f}s")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "wall_clock_seconds": wall_clock_seconds,
        "cpu_seconds": cpu_time_seconds,
        "gpu_seconds": gpu_time_seconds,
        "question_results": question_results
    }


def parse_bool(value):
    """Convert a string or bool input to a boolean"""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes"):
            return True
        if normalized in ("false", "0", "no"):
            return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MMLU Evaluation with timing and comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  MODEL_NAME         Comma-separated list of models (default: meta-llama/Llama-3.2-1B-Instruct)
  QUICK_TEST         Run on first 2 subjects only (True/False, default: False)
  VERBOSE_OUTPUT     Save detailed Q&A to file (True/False, default: False)
  QUANTIZATION_BITS  Use 4 or 8 for quantization (default: None)
  USE_GPU            Use GPU if available (True/False, default: True)

CLI arguments override environment variables when provided.

Examples:
  python llama_mmlu_eval.py --verbose --quick-test
  python llama_mmlu_eval.py --model-name "meta-llama/Llama-3.2-1B-Instruct,microsoft/Phi-3-mini-4k-instruct"
  python llama_mmlu_eval.py --use-gpu False
  export MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct,microsoft/Phi-3-mini-4k-instruct"
  python llama_mmlu_eval.py
        """
    )
    parser.add_argument('--verbose', action='store_true',
                       help='Save each question and answer to verbose output file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run on first 2 subjects only (overrides QUICK_TEST env var)')
    parser.add_argument('--model-name',
                       help='Comma-separated list of models (overrides MODEL_NAME env var)')
    parser.add_argument('--use-gpu', type=parse_bool,
                       help='Use GPU if available (True/False, overrides USE_GPU env var)')
    parser.add_argument('--quantization-bits', type=int,
                       choices=[4, 8],
                       help='Use 4 or 8 for quantization (overrides QUANTIZATION_BITS env var)')
    return parser.parse_args()


def main():
    """Main evaluation function"""
    # Setup output directories
    setup_output_directories()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply CLI argument overrides to global variables
    global QUICK_TEST, VERBOSE_OUTPUT, MMLU_SUBJECTS, MODEL_NAMES, USE_GPU, QUANTIZATION_BITS
    if args.quick_test:
        QUICK_TEST = True
    if args.verbose:
        VERBOSE_OUTPUT = True
    if args.model_name:
        model_names_override = [name.strip() for name in args.model_name.split(",") if name.strip()]
        if not model_names_override:
            print("❌ Error: --model-name provided but no valid model names were found")
            sys.exit(1)
        MODEL_NAMES = model_names_override
    if args.use_gpu is not None:
        USE_GPU = args.use_gpu
    if args.quantization_bits is not None:
        QUANTIZATION_BITS = args.quantization_bits
    
    # Apply subject filtering based on QUICK_TEST
    if QUICK_TEST:
        MMLU_SUBJECTS = MMLU_SUBJECTS[:2]
        print(f"🚀 QUICK TEST MODE: Using first {len(MMLU_SUBJECTS)} subjects only")
    else:
        MMLU_SUBJECTS = MMLU_SUBJECTS[:10]
        print(f"📚 Using first {len(MMLU_SUBJECTS)} subjects")
    
    print("\n" + "="*70)
    print("Llama MMLU Evaluation (Quantized)")
    print("="*70 + "\n")

    # Check environment
    in_colab, device = check_environment()

    # Store results for all models
    all_models_results = []
    overall_start_time = datetime.now()
    
    # Evaluate each model
    for model_idx, model_name in enumerate(MODEL_NAMES, 1):
        print(f"\n{'='*70}")
        print(f"Model {model_idx}/{len(MODEL_NAMES)}: {model_name}")
        print(f"{'='*70}\n")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        # Evaluate
        results = []
        total_correct = 0
        total_questions = 0
        
        print(f"\n{'='*70}")
        print(f"Starting evaluation on {len(MMLU_SUBJECTS)} subjects")
        print(f"{'='*70}\n")
        
        # Open verbose output file if enabled
        verbose_file = None
        if VERBOSE_OUTPUT:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_safe_name = model_name.replace("/", "_").replace(".", "_")
            verbose_filename = os.path.join(OUTPUT_DIRS["logs"], f"verbose_{model_safe_name}_{timestamp}.txt")
            verbose_file = open(verbose_filename, "w")
            verbose_file.write(f"{'='*70}\n")
            verbose_file.write(f"MMLU Evaluation Verbose Output\n")
            verbose_file.write(f"Model: {model_name}\n")
            verbose_file.write(f"Timestamp: {timestamp}\n")
            verbose_file.write(f"{'='*70}\n\n")
            print(f"📝 Verbose output will be saved to: {verbose_filename}")
        
        # Start model-level timing
        model_start_time = datetime.now()
        
        for i, subject in enumerate(MMLU_SUBJECTS, 1):
            print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
            result = evaluate_subject(
                model,
                tokenizer,
                subject,
                verbose_output=VERBOSE_OUTPUT,
                verbose_file_handle=verbose_file
            )
            if result:
                results.append(result)
                total_correct += result["correct"]
                total_questions += result["total"]
        
        # End model-level timing
        model_end_time = datetime.now()
        model_wall_clock = (model_end_time - model_start_time).total_seconds()
        model_cpu_time = sum(result["cpu_seconds"] for result in results)
        model_gpu_time = sum(result["gpu_seconds"] for result in results)
        
        # Close verbose file
        if verbose_file:
            verbose_file.write(f"\n{'='*70}\n")
            verbose_file.write(f"Evaluation Complete\n")
            verbose_file.write(f"{'='*70}\n")
            verbose_file.close()
            print(f"✓ Verbose output saved to: {verbose_filename}")
        
        # Calculate overall accuracy
        overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
        
        # Print summary for this model
        print("\n" + "="*70)
        print(f"EVALUATION SUMMARY - {model_name}")
        print("="*70)
        print(f"Total Subjects: {len(results)}")
        print(f"Total Questions: {total_questions}")
        print(f"Total Correct: {total_correct}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print(f"\nTiming Breakdown:")
        print(f"  Wall Clock: {model_wall_clock:.2f}s ({model_wall_clock/60:.1f} min)")
        print(f"  CPU Time:   {model_cpu_time:.2f}s")
        print(f"  GPU Time:   {model_gpu_time:.2f}s")
        print("="*70)
        
        # Store results for this model
        all_models_results.append({
            "model_name": model_name,
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "timing": {
                "wall_clock_seconds": model_wall_clock,
                "cpu_seconds": model_cpu_time,
                "gpu_seconds": model_gpu_time,
            },
            "subject_results": results,
            "top_5_subjects": sorted(results, key=lambda x: x["accuracy"], reverse=True)[:5],
            "bottom_5_subjects": sorted(results, key=lambda x: x["accuracy"], reverse=True)[-5:]
        })
        
        # Save per-model results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quant_suffix = f"_{QUANTIZATION_BITS}bit" if QUANTIZATION_BITS else "_full"
        model_safe_name = model_name.replace("/", "_").replace(".", "_")
        output_file = os.path.join(OUTPUT_DIRS["evals"], f"llama_mmlu_results_{model_safe_name}{quant_suffix}_{timestamp}.json")
        
        output_data = {
            "model": model_name,
            "quantization_bits": QUANTIZATION_BITS,
            "timestamp": timestamp,
            "device": str(device),
            "timing": {
                "wall_clock_seconds": model_wall_clock,
                "cpu_seconds": model_cpu_time,
                "gpu_seconds": model_gpu_time,
            },
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "subject_results": results
        }
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Print top/bottom subjects
        if len(results) > 0:
            sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
            
            print("\n📊 Top 5 Subjects:")
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")
            
            print("\n📉 Bottom 5 Subjects:")
            for i, result in enumerate(sorted_results[-5:], 1):
                print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")
        
        # Cleanup between models
        del model, tokenizer
        torch.cuda.empty_cache()
        
        if model_idx < len(MODEL_NAMES):
            print("\n" + "="*70)
            print("Cleaning up GPU memory...")
            print("="*70)
    
    overall_end_time = datetime.now()
    overall_duration = (overall_end_time - overall_start_time).total_seconds()
    
    # Save comparison results
    if len(all_models_results) > 1:
        print("\n" + "="*70)
        print("COMPARISON ACROSS ALL MODELS")
        print("="*70)
        
        comparison_data = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "total_duration_seconds": overall_duration,
            "total_models": len(MODEL_NAMES),
            "quantization_bits": QUANTIZATION_BITS,
            "device": str(device),
            "models": all_models_results
        }
        
        # Print comparison table
        print(f"\n{'Model':<40} {'Accuracy':<12} {'Wall(s)':<10} {'CPU(s)':<10} {'GPU(s)':<10}")
        print("-" * 82)
        for result in all_models_results:
            timing = result['timing']
            print(f"{result['model_name']:<40} "
                  f"{result['overall_accuracy']:<12.2f}% "
                  f"{timing['wall_clock_seconds']:<10.1f} "
                  f"{timing['cpu_seconds']:<10.1f} "
                  f"{timing['gpu_seconds']:<10.1f}")
        
        # Save comparison to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quant_suffix = f"_{QUANTIZATION_BITS}bit" if QUANTIZATION_BITS else "_full"
        comparison_file = os.path.join(OUTPUT_DIRS["evals"], f"llama_mmlu_comparison{quant_suffix}_{timestamp}.json")
        
        with open(comparison_file, "w") as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\n✓ Comparison saved to: {comparison_file}")
    
    # Generate comparison graphs (works for 1 or more models)
    if len(all_models_results) >= 1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quant_suffix = f"_{QUANTIZATION_BITS}bit" if QUANTIZATION_BITS else "_full"
        create_comparison_graphs(all_models_results, 
                                output_prefix=os.path.join(OUTPUT_DIRS["images"], f"mmlu_comparison{quant_suffix}_{timestamp}"))
        
        # Generate missed questions histogram for multi-model runs
        if len(all_models_results) > 1:
            create_missed_questions_histogram(all_models_results, MODEL_NAMES,
                                            output_path=os.path.join(OUTPUT_DIRS["images"], f"mmlu_missed_questions{quant_suffix}_{timestamp}.png"))
    
    # Colab-specific instructions
    if in_colab:
        print("\n" + "="*70)
        print("💾 To download results in Colab:")
        print("="*70)
        print(f"from google.colab import files")
        print(f"files.download('<filename>')")
    
    print("\n✅ All evaluations complete!")
    print(f"Total time: {overall_duration/60:.1f} minutes")
    return all_models_results


def create_missed_questions_histogram(all_models_results, model_names, output_path="mmlu_missed_questions.png"):
    """Generate histogram showing how many models missed each question"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping missed questions histogram.")
        print("   Install with: pip install matplotlib")
        return
    
    if len(all_models_results) < 2:
        print("\n⚠️  Need at least 2 models to generate missed questions histogram")
        return
    
    print(f"\n{'='*70}")
    print("📊 Generating missed questions histogram...")
    print(f"{'='*70}\n")
    
    # Accumulate question results across all models
    all_question_data = defaultdict(list)
    
    for model_result in all_models_results:
        model_name = model_result['model_name']
        subject_results = model_result['subject_results']
        
        for subject_result in subject_results:
            if 'question_results' in subject_result:
                for question_hash, is_correct in subject_result['question_results'].items():
                    all_question_data[question_hash].append(is_correct)
    
    # Count how many models missed each question
    missed_by_counts = defaultdict(int)
    total_unique_questions = 0
    
    for question_hash, results in all_question_data.items():
        total_unique_questions += 1
        num_models_missed = sum(1 for correct in results if not correct)
        missed_by_counts[num_models_missed] += 1
    
    # Create histogram data
    num_models = len(all_models_results)
    x_values = list(range(0, num_models + 1))  # 0 to N models missed
    y_values = [missed_by_counts.get(x, 0) for x in x_values]
    
    # Calculate statistics
    total_missed = sum(y_values[1:])  # Exclude 0 (all correct)
    total_misses = sum(num_missed * count for num_missed, count in missed_by_counts.items())
    avg_missed = total_misses / total_unique_questions if total_unique_questions > 0 else 0
    hardest_questions = y_values[-1]  # Questions all models missed
    easiest_questions = y_values[0]  # Questions all models got right
    
    print(f"Statistics:")
    print(f"  Total unique questions: {total_unique_questions}")
    print(f"  Questions all models got right: {easiest_questions} ({easiest_questions/total_unique_questions*100:.1f}%)")
    print(f"  Questions all models missed: {hardest_questions} ({hardest_questions/total_unique_questions*100:.1f}%)")
    print(f"  Average misses per question: {avg_missed:.2f}")
    print()
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    bars = plt.bar(x_values, y_values, color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Number of Models That Missed the Question', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Questions', fontsize=12, fontweight='bold')
    plt.title('Distribution of Question Difficulty Across Models', fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add statistics text box
    stats_text = f'Total Questions: {total_unique_questions}\n'
    stats_text += f'Models: {num_models}\n'
    stats_text += f'All Got Right: {easiest_questions} ({easiest_questions/total_unique_questions*100:.1f}%)\n'
    stats_text += f'All Missed: {hardest_questions} ({hardest_questions/total_unique_questions*100:.1f}%)'
    
    plt.text(0.98, 0.98, stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Set x-axis ticks to show model numbers
    plt.xticks(x_values, [str(x) for x in x_values])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    print(f"\n📊 Missed questions histogram generated successfully!")


def create_comparison_graphs(all_models_results, output_prefix="mmlu_comparison"):
    """Generate comparison graphs for model evaluation results"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n⚠️  matplotlib not installed. Skipping graph generation.")
        print("   Install with: pip install matplotlib")
        return
    
    if len(all_models_results) == 0:
        print("\n⚠️  No results to graph")
        return
    
    print(f"\n{'='*70}")
    print("📊 Generating comparison graphs...")
    print(f"{'='*70}\n")
    
    # Extract data
    model_names = [r['model_name'].split('/')[-1] for r in all_models_results]  # Short names
    accuracies = [r['overall_accuracy'] for r in all_models_results]
    wall_times = [r['timing']['wall_clock_seconds'] for r in all_models_results]
    cpu_times = [r['timing']['cpu_seconds'] for r in all_models_results]
    gpu_times = [r['timing']['gpu_seconds'] for r in all_models_results]
    
    # Define color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    # Graph 1: Model Accuracy Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('MMLU Evaluation: Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    accuracy_file = f'{output_prefix}_accuracy.png'
    plt.savefig(accuracy_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {accuracy_file}")
    plt.close()
    
    # Graph 2: Timing Comparison (Grouped Bar Chart)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, wall_times, width, label='Wall Clock', color='#3498db')
    bars2 = ax.bar(x, cpu_times, width, label='CPU Time', color='#e74c3c')
    bars3 = ax.bar(x + width, gpu_times, width, label='GPU Time', color='#2ecc71')
    
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('MMLU Evaluation: Timing Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there's actual time
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    timing_file = f'{output_prefix}_timing.png'
    plt.savefig(timing_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {timing_file}")
    plt.close()
    
    print(f"\n📊 Graphs generated successfully!")


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
