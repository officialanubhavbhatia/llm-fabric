from myvista import MyVista

client = MyVista()
client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello"}],
)
traces = client.traces.list()
print(traces["scope"], len(traces["traces"]))
if traces["traces"]:
    print(client.traces.get(traces["traces"][0]["trace_id"])["trace_id"])
