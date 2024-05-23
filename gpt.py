from g4f.client import Client

content = "张量在机器学习中的主要用途"


client = Client()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": content}],
)
print(response.choices[0].message.content)