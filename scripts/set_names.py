import json

filename = input("Enter input ordering file: ")

with open(filename, "r") as f:
    data = json.loads(f.read())
    print("Note: if you are using an X-LoRA model, it is very important that the adapter names are specified in the correct order"
          ", which is the order used during training. If you are using a LoRA model this is not necessary.")
    adapters = input(
        "Enter a comma delimited list of adapter names: "
    )
    split = adapters.split(",")
    split = [x for x in split if len(x) > 0]
    split = [x.strip() for x in split]
    data["order"] = split
    outname = input("Enter output ordering file: ")
    with open(outname, "w") as f:
        f.write(json.dumps(data))
