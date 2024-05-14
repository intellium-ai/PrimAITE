from text_generation import Client

client = Client("http://192.168.0.8:58084")

resp = client.generate(prompt="Hello, ", max_new_tokens=4096)

print(resp.generated_text)
