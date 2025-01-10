import sglang as sgl

# Initialize the model
gen = sgl.Runtime(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",
    quantization="awq",
    tensor_parallel_size=4
)

# Define a simple function
@sgl.function
def generate(state, prompt):
    state += sgl.text(prompt)
    state += sgl.gen(max_tokens=1024)

# Run inference
result = generate("Write about AI")