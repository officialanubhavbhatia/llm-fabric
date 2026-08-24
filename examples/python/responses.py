from myvista import MyVista

client = MyVista()
response = client.responses.create(input="Debug this Python program")
print(response.output_text)
