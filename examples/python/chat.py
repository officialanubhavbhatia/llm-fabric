from myvista import MyVista

client = MyVista()
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.text)
print(response.request_id, response.fabric.served_model)
