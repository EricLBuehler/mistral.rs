# Generate a UQFF Hugging Face model card .md file.

msg = "This script is used to generate a Hugging Face model card."
print("-" * len(msg))
print(msg)
print("-" * len(msg))

model_id = input("Please enter the original model ID: ")
display_model_id = input(
    "Please enter the model ID where this model card will be displayed: "
)

output = f"""---
tags:
  - uqff
  - mistral.rs
base_model: {model_id}
base_model_relation: quantized
---

<!-- Autogenerated from user input. -->

"""


output += f"# `{model_id}`, UQFF quantization\n\n"

output += """
Run with [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Documentation: [UQFF docs](https://github.com/EricLBuehler/mistral.rs/blob/master/docs/UQFF.md).

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Reliable** 🔒: Compatibility ensured with *embedded* and *checked* semantic versioning information from day 1.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
3) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.
"""

print(" NOTE: Getting metadata now, press CTRL-C when you have entered all files")
print(
    " NOTE: If multiple quantizations were used: enter the quantization names, and then in the next prompt, the topology file used."
)

output += f"## Files\n\n"

output += "|Name|Quantization type(s)|Example|\n|--|--|--|\n"

topologies = {}

try:
    n = 0
    while True:
        print(
            f" NOTE: Next file. Have processed {n} files. Press CTRL-C now if there are no more."
        )
        file = input("Enter UQFF filename (with extension): ").strip()
        output += f"|{file}|"

        quants = input(
            "Enter quantization NAMES used to make that file (single quantization name, OR if multiple, comma delimited): "
        )
        if "," in quants:
            quants = list(map(lambda x: x.strip().upper(), quants.split(",")))
            topology = input(
                "Enter topology used to make UQFF with multiple quantizations: "
            )
            topologies[file] = topology
            output += f"{",".join(quants)} (see topology for this file)|"
        else:
            output += f"{quants.strip().upper()}|"
        # This interactive mode only will work for text models...
        output += f"`./mistralrs-server -i plain -m {model_id} --from-uqff {display_model_id}/{file}`|\n"
        n += 1
        print()
except KeyboardInterrupt:
    pass

if n == 0:
    raise ValueError("Need at least one file")

if len(topologies):
    output += "\n\n## Topologies\n**The following model topologies were used to generate this UQFF file. Only information pertaining to ISQ is relevant.**\n"
    for name, file in topologies.items():
        with open(file, "r") as f:
            output += f"### Used for `{name}`\n\n"
            output += f"```yml\n{f.read()}\n```\n"

msg = "Done! Please enter the output filename"
print("\n" + "-" * len(msg))
print(msg)
print("-" * len(msg))

out = input("Enter the output filename: ")
with open(out, "a") as f:
    f.write(output)
