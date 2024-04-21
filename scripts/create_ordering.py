from peft.tuners import lora
from transformers import AutoModelForCausalLM  # type: ignore
import json
from peft.tuners.lora.config import LoraConfig

model_id = input("Enter the base model ID: ")
target_modules_in = input("Enter the target modules as a comma delimited list: ")
target_modules = target_modules_in.split(",")
target_modules = [x for x in target_modules if len(x) > 0]
target_modules = [x.strip() for x in target_modules]

model = AutoModelForCausalLM.from_pretrained(model_id)
lora_config = LoraConfig(target_modules=target_modules, init_lora_weights=False)

model.add_adapter(lora_config, "default")

total_swapped = 0
loras = {}
for n, module in model.named_modules():
    if isinstance(module, lora.Linear):
        loras[n.split("lora_A.")[0]] = total_swapped
        total_swapped += 1
    elif isinstance(module, lora.Embedding):
        loras[n.split("lora_embedding_A.")[0]] = total_swapped
        total_swapped += 1
    elif isinstance(module, lora.Conv2d):
        loras[n.split("lora_A.")[0]] = total_swapped
        total_swapped += 1

adapters_in = input(
    "Enter a comma delimited list of adapter names as they were specified when training: "
)
adapters = adapters_in.split(",")
adapters = [x for x in adapters if len(x) > 0]
adapters = [x.strip() for x in adapters]

out = {"order": adapters, "layers": loras, "base_model_id": model_id}

outfile = input("Enter output file: ")
with open(outfile, "w") as f:
    f.write(json.dumps(out))
