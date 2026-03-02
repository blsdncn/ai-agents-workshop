from openai import OpenAI
import getpass, os

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Say: Working!"}],
    max_completion_tokens=512,
)

print(f"✓ Success! Response: {response.choices[0].message.content}")
print(f"Cost: ${response.usage.total_tokens * 0.000000375:.6f}")
