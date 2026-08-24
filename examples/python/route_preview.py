from myvista import MyVista

client = MyVista()
plan = client.routes.preview(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
print(plan["explanation"])
