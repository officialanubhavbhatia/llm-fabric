from myvista import MyVista

client = MyVista()
# `response_format` is accepted by the OpenAI dialect and ignored by this
# fabric. Ask for JSON in the prompt; the gateway does not validate a schema.
response = client.chat.completions.create(
    model="auto",
    messages=[
        {
            "role": "user",
            "content": 'Reply with JSON only: {"ok": true}',
        }
    ],
    response_format={"type": "json_object"},
)
print(response.text)
