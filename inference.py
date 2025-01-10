from vllm import LLM, SamplingParams
import time
import os


# Get HF token from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable")

os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token  # Set for HuggingFace library


# Configure model settings
model_name = "Qwen/Qwen-32B"
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    presence_penalty=0,
    frequency_penalty=0
)

# Initialize LLM with 4-bit quantization
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,  # Use all 4 GPUs
    quantization="awq",      # 4-bit AWQ quantization
    trust_remote_code=True,
    max_model_len=8192,
    dtype="float16"         # Mix with fp16 for efficiency
)

# Test prompt
prompt = "Write a detailed essay about artificial intelligence." * 32  # To get roughly 1024 tokens

# Measure generation speed
start_time = time.time()
outputs = llm.generate([prompt], sampling_params)
end_time = time.time()

# Calculate tokens per second
output_text = outputs[0].outputs[0].text
input_tokens = len(llm.get_tokenizer().encode(prompt))
output_tokens = len(llm.get_tokenizer().encode(output_text))
total_time = end_time - start_time
tokens_per_second = output_tokens / total_time

print(f"Tokens per second: {tokens_per_second}")
print(f"Total generation time: {total_time} seconds")
print(f"Output tokens: {output_tokens}")