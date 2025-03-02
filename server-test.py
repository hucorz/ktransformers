from openai import OpenAI
client = OpenAI(base_url="http://localhost:10002/v1", api_key="xxx")

completion = client.chat.completions.create(
  model="/fs/fast/u20247643/hf/models/Qwen2.5-7B-Instruct",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
