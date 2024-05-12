import json

filename = input("Enter input ordering file: ")

with open(filename, "r") as f:
    data = json.loads(f.read())
    preload_adapters = input(
        "Enter a comma delimited list of *preloaded* adapter names to preload: "
    )
    preload_adapters_model_id = input(
        "Enter the model id where the preload adapters will be loaded: "
    )
    split = preload_adapters.split(",")
    split = [x for x in split if len(x) > 0]
    split = [x.strip() for x in split]
    res = []
    for s in split:
        res.append({"name": s, "adapter_model_id": preload_adapters_model_id})
    data["preload_adapters"] = res
    outname = input("Enter output ordering file: ")
    with open(outname, "w") as f:
        f.write(json.dumps(data))
