from myvista import MyVista

client = MyVista()
run = client.evals.run("ci")
print(run["suite_name"], run["metrics"])
