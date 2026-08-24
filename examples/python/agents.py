from myvista import MyVista, UnsupportedError

client = MyVista()
try:
    client.agents.run(input="book a flight")
except UnsupportedError as exc:
    print(exc)
