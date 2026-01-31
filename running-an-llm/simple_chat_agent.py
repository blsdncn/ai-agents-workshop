"""
Bare-Bones Chat Agent for Llama 3.2-1B-Instruct

This is a minimal chat interface that demonstrates:
1. How to load a model without quantization
2. How chat history is maintained and fed back to the model
3. The difference between plain text history and tokenized input

No classes, no fancy features - just the essentials.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# CONFIGURATION - Change these settings as needed
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# System prompt - This sets the chatbot's behavior and personality
# Change this to customize how the chatbot responds
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."

# Flag to disable conversation history (stateless mode)
# If True, only the system prompt and the latest user input are sent to the model
DISABLE_HISTORY = os.environ.get("DISABLE_HISTORY", "False").lower() in ('true', '1', 'yes')

SUMMARIZATION_PROMPT = "Summarize the previous provided conversation history in up to 2000 words. Capture high level objective of conversation and key details."

MAX_TOTAL_TOKENS = 8192

SUMMARY_THRESHOLD = 6000

MAX_INPUT_TOKENS = 1024

MAX_OUTPUT_TOKENS = 1024

MAX_SUMMARY_OUTPUT = 2000

# ============================================================================
# LOAD MODEL (NO QUANTIZATION)
# ============================================================================

print("Loading model (this takes 1-2 minutes)...")

# Load tokenizer (converts text to numbers and vice versa)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in half precision (float16) for efficiency
# Use float16 on GPU, or float32 on CPU if needed
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,                        # Use FP16 for efficiency
    device_map="auto",                          # Automatically choose GPU/CPU
    low_cpu_mem_usage=True
)

model.eval()  # Set to evaluation mode (no training)
print(f"✓ Model loaded! Using device: {model.device}")
print(f"✓ Memory usage: ~2.5 GB (FP16)\n")

if DISABLE_HISTORY:
    print("⚠️ Conversation history is DISABLED (Stateless mode)")
    print("Each turn will only include the system prompt and the current user input.\n")

# ============================================================================
# CHAT HISTORY - This is stored as PLAIN TEXT (list of dictionaries)
# ============================================================================
# The chat history is a list of messages in this format:
# [
#   {"role": "system", "content": "You are a helpful assistant"},
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi! How can I help?"},
#   {"role": "user", "content": "What's 2+2?"},
#   {"role": "assistant", "content": "2+2 equals 4."}
# ]
#
# This is PLAIN TEXT - humans can read it
# The model CANNOT use this directly - it needs to be tokenized first

chat_history = []

# Add system prompt to history (this persists across the entire conversation)
chat_history.append({
    "role": "system",
    "content": SYSTEM_PROMPT
})


# ============================================================================
# CONTEXT MANAGEMENT FUNCTIONS
# ============================================================================

def count_tokens(messages):
    """
    Count the number of tokens in the chat history.

    TODO: Implement this function
    - Use tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
    - Return the length of the resulting token tensor
    - This should count template overhead tokens (special tags, etc.)
    """
    return tokenizer.apply_chat_template(messages,
                                  tokenize=True,
                                  add_generation_prompt=True,
                                  return_tensors="pt").numel()




def get_summary(history_to_summarize):
    """
    Generate a summary of the provided conversation history.

    TODO: Implement this function
    - Create a summary prompt asking the model to condense the history
    - Tokenize the prompt and pass it to the model
    - Generate a response with max_new_tokens=MAX_SUMMARY_OUTPUT
    - Return the generated summary text
    - Use do_sample=False for deterministic summarization
    """
    input_ids = tokenizer.apply_chat_template(
           [{"role": "system", "content": SUMMARIZATION_PROMPT}] + history_to_summarize,  # Our PLAIN TEXT history
        add_generation_prompt=True,       # Add prompt for assistant's response
        return_tensors="pt",              # Return as PyTorch tensor (numbers)
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)


    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=MAX_SUMMARY_OUTPUT,              # Maximum length of response
            do_sample=False,                  # Use sampling for variety
            temperature=0.5,                 # Lower = more focused, higher = more random
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract only the newly generated tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        assistant_response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True  # Remove special tokens like <|end|>
        )
        return f"[Summarized chat history] {assistant_response}"



    



def manage_context(chat_history):
    """
    Check if context exceeds threshold and apply hybrid buffer strategy.

    Strategy:
    1. Always keep system prompt (index 0)
    2. Keep last 5 messages (2 full exchanges + current user input)
    3. Summarize everything in between
    4. If history is too short for summarization (< 7 messages), truncate the longest message
    """
    current_tokens = count_tokens(chat_history)
    if current_tokens > SUMMARY_THRESHOLD:
        print("Hit summary threshold.")
        # We need: System (1) + at least 1 message to summarize + Recent (5) = 7 messages
        if len(chat_history) >= 7:
            print("📊 Context threshold exceeded. Summarizing older history...")
            summary_content = get_summary(chat_history[1:-5])
            new_history = [chat_history[0]]
            new_history.append({
                "role": "user",
                "content": summary_content
            })
            new_history.extend(chat_history[-5:])
            return new_history
        else:
            # History too short to summarize effectively, truncate longest message
            print("⚠️ Context threshold exceeded but history too short to summarize. Truncating longest message...")
            longest_idx = -1
            max_len = -1
            # Start from 1 to avoid truncating system prompt
            for i in range(1, len(chat_history)):
                msg_len = len(chat_history[i]["content"])
                if msg_len > max_len:
                    max_len = msg_len
                    longest_idx = i
            
            if longest_idx != -1:
                content = chat_history[longest_idx]["content"]
                # Truncate to 25% of original length to aggressively clear space
                truncated_content = content[:len(content)//4] + "... [TRUNCATED]"
                chat_history[longest_idx]["content"] = truncated_content
    #print(f"Current history length: {len(chat_history)} | Current tokens: {count_tokens(chat_history)}")
    return chat_history



# ============================================================================
# CHAT LOOP
# ============================================================================

print("="*70)
print("Chat started! Type 'quit' or 'exit' to end the conversation.")
print("="*70 + "\n")

while True:
    # ========================================================================
    # STEP 1: Get user input (PLAIN TEXT)
    # ========================================================================
    user_input = input("You: ").strip()
    
    # Check for exit commands
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nGoodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue


    user_input_history_entry = {
            "role": "user",
            "content": user_input
            }

    if count_tokens([user_input_history_entry]) > MAX_INPUT_TOKENS:
        print("\nInput too long.")
        continue

    
    # ========================================================================
    # STEP 2: Add user message to chat history (PLAIN TEXT)
    # ========================================================================
    # The chat history grows with each exchange
    # We append the new user message to the existing history
    chat_history.append(user_input_history_entry)




    # ========================================================================
    # CONTEXT MANAGEMENT
    # ========================================================================
    # Check if we need to summarize history to stay within token limits
    # This happens BEFORE tokenization to ensure we always feed a managed context

    chat_history = manage_context(chat_history)

    # At this point, chat_history looks like:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},      ← Just added
    # ]
    # This is still PLAIN TEXT
    
    # ========================================================================
    # STEP 3: Convert chat history to model input (TOKENIZATION)
    # ========================================================================
    # The model needs numbers (tokens), not text
    # apply_chat_template() does two things:
    #   1. Formats the chat history with special tokens (like <|start|>, <|end|>)
    #   2. Converts the formatted text into token IDs (numbers)
    
    # First, apply_chat_template formats the history and converts to tokens
    
    # If history is disabled, we only use the system prompt and current user message
    if DISABLE_HISTORY:
        messages_to_send = [chat_history[0], chat_history[-1]]
    else:
        messages_to_send = chat_history

    input_ids = tokenizer.apply_chat_template(
        messages_to_send,                # Managed chat history
        add_generation_prompt=True,      # Add prompt for assistant's response
        return_tensors="pt"              # Return as PyTorch tensor (numbers)
    ).to(model.device)

    # Create attention mask (1 for all tokens since we have no padding)
    attention_mask = torch.ones_like(input_ids)

    # Now input_ids is TOKENIZED - it's a tensor of numbers like:
    # tensor([[128000, 128006, 9125, 128007, 271, 2675, 527, 264, ...]])
    # These numbers represent our entire conversation history

    # ========================================================================
    # STEP 4: Generate assistant response (MODEL INFERENCE)
    # ========================================================================
    # The model looks at the ENTIRE chat history (in tokenized form)
    # and generates a response

    print("Assistant: ", end="", flush=True)

    with torch.no_grad():  # Don't calculate gradients (we're not training)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,   # Explicitly pass attention mask
            max_new_tokens=MAX_OUTPUT_TOKENS, # Maximum length of response
            do_sample=True,                  # Use sampling for variety
            temperature=0.7,                 # Lower = more focused, higher = more random
            top_p=0.9,                       # Nucleus sampling
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs contains: [original input tokens + new generated tokens]
    # We only want the NEW tokens (the assistant's response)
    
    # ========================================================================
    # STEP 5: Decode the response (DETOKENIZATION)
    # ========================================================================
    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    
    # Convert tokens (numbers) back to text (PLAIN TEXT)
    assistant_response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True  # Remove special tokens like <|end|>
    )
    
    print(assistant_response)  # Display the response
    
    # ========================================================================
    # STEP 6: Add assistant response to chat history (PLAIN TEXT)
    # ========================================================================
    # This is crucial! We add the assistant's response to the history
    # so the model remembers what it said in future turns
    
    chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })
    
    # Now chat_history has grown again:
    # [
    #   {"role": "system", "content": "You are helpful..."},
    #   {"role": "user", "content": "Hello!"},
    #   {"role": "assistant", "content": "Hi!"},
    #   {"role": "user", "content": "What's 2+2?"},
    #   {"role": "assistant", "content": "4"}              ← Just added
    # ]
    
    # When the loop repeats:
    # - User enters new message
    # - We add it to chat_history
    # - We tokenize the ENTIRE history (including all previous exchanges)
    # - Model sees everything and generates response
    # - We add response to history
    # - Repeat...
    
    # This is how the chatbot "remembers" the conversation!
    # Each turn, we feed it the ENTIRE conversation history
    
    print()  # Blank line for readability

# ============================================================================
# SUMMARY OF HOW CHAT HISTORY WORKS
# ============================================================================
"""
PLAIN TEXT vs TOKENIZED:

1. PLAIN TEXT (chat_history):
   - Human-readable format
   - List of dictionaries: [{"role": "user", "content": "Hi"}, ...]
   - Stored in memory between turns
   - Gets longer with each message

2. TOKENIZED (input_ids):
   - Numbers (token IDs)
   - Created fresh each turn from chat_history
   - This is what the model actually "reads"
   - Example: [128000, 128006, 9125, 128007, ...]

PROCESS EACH TURN:
   User input (text)
   ↓
   Add to chat_history (text)
   ↓
   Tokenize entire chat_history (text → numbers)
   ↓
   Model generates response (numbers)
   ↓
   Decode response (numbers → text)
   ↓
   Add response to chat_history (text)
   ↓
   Loop back to start

WHY FEED ENTIRE HISTORY?
- The model has no memory between calls
- Each generation is independent
- To "remember" previous turns, we must include them in the input
- This is why context length matters - longer conversations = more tokens

WHAT HAPPENS AS CONVERSATION GROWS?
- chat_history gets longer (more messages)
- Tokenized input gets longer (more tokens)
- Eventually hits model's max context length (for Llama 3.2: 128K tokens)
- Then you need context management (truncation, summarization, etc.)
- But for this simple demo, we let it grow without limit
"""
