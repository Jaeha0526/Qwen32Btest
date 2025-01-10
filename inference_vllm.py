from vllm import LLM, SamplingParams
import time
import os


# Get HF token from environment variable
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    raise ValueError("Please set HF_TOKEN environment variable")

os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token  # Set for HuggingFace library

# Set hugginface base in workspace
os.environ['HF_HOME'] = '/workspace/huggingface'
os.environ['HUGGING_FACE_HUB_CACHE'] = '/workspace/huggingface/hub'
os.environ["TRANSFORMERS_CACHE"] = "/workspace/huggingface/hub"

# Configure model settings
sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    max_tokens=1024,
    presence_penalty=0,
    frequency_penalty=0
)

# Initialize LLM with AWQ quantization
llm = LLM(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",
    tensor_parallel_size=4,
    quantization="awq",
    trust_remote_code=True,
    max_model_len=8192,
    dtype="float16"
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
print(f"Output: {output_text}")