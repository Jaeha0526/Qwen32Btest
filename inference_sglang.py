import sglang as sgl
import time
import os

# Initialize the model with the same config as vLLM script
gen = sgl.Runtime(
    model="Qwen/Qwen2-72B-Instruct-AWQ",
    quantization="awq",
    tensor_parallel_size=4,
    trust_remote_code=True,
    max_model_len=8192,
    dtype="float16"
)

# Define a simple function
@sgl.function
def generate(state, prompt):
    state = sgl.State()
    state += sgl.text(prompt)
    state += sgl.gen("response", max_tokens=1024, temperature=1.0, top_p=1.0)
    return state["response"]

# Test prompt
prompt = "Write a short essay about artificial intelligence."

# Measure generation speed
start_time = time.time()
result = generate(prompt)
end_time = time.time()

# Print results
total_time = end_time - start_time
print(f"\nTotal generation time: {total_time:.2f} seconds")
print(f"\nGenerated text:\n{result}")