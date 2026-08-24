from myvista import MyVista, UnsupportedError

client = MyVista()
try:
    client.embeddings.create(input="hello", model="auto")
except UnsupportedError as exc:
    print(exc)
