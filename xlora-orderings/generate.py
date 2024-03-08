import json

filename = input("Enter input ordering file: ")

with open(filename, "r") as f:
    data = json.loads(f.read())
    adapters = input("Enter a comma delimited list of adapter names as they were specified when training: ")
    split = adapters.split(",")
    split = [x for x in split if len(x) > 0]
    split = [x.strip() for x in split]
    data["order"] = split
    outname = input("Enter output ordering file: ")
    with open(outname, "w") as f:
        f.write(json.dumps(data))