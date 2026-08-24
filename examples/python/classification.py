from myvista import MyVista

client = MyVista()
decision = client.intents.classify("debug this python traceback")
print(decision["classification"]["intent_id"], decision["classification"]["confidence"])
