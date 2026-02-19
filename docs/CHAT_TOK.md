# Chat templates and tokenizer customization

## JINJA chat templates (recommended method)
Some models do not come with support for tool calling or other features, and as such it might be necessary to specify your own chat template.

We provide some chat templates [here](https://github.com/EricLBuehler/mistral.rs/tree/master/chat_templates/), and it is easy to modify or create others to customize chat template behavior.

To use this, add the `jinja-explicit` parameter to the various APIs

```bash
mistralrs serve -p 1234 --isq 4 --jinja-explicit chat_templates/mistral_small_tool_call.jinja -m mistralai/Mistral-Small-3.1-24B-Instruct-2503
```

## Chat template overrides
Mistral.rs attempts to automatically load a chat template from the `tokenizer_config.json` file. This enables high flexibility across instruction-tuned models and ensures accurate chat templating. However, if the `chat_template` field is missing, then a JINJA chat template should be provided. The JINJA chat template may use `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.

We provide some chat templates [here](https://github.com/EricLBuehler/mistral.rs/tree/master/chat_templates/), and it is easy to modify or create others to customize chat template behavior.

For example, to use the `chatml` template, `--chat-template` is specified *before* the model architecture. For example:

```bash
mistralrs serve -p 1234 --log output.log --chat-template ./chat_templates/chatml.json -m meta-llama/Llama-3.2-3B-Instruct
```

> Note: For GGUF models, the chat template may be loaded directly from the GGUF file by omitting any other chat template sources.

## Tokenizer

Some models do not provide a `tokenizer.json` file although mistral.rs expects one. To solve this, please run [this](https://github.com/EricLBuehler/mistral.rs/blob/master/scripts/get_tokenizers_json.py) script. It will output the `tokenizer.json` file for your specific model. This may be used by passing the `--tokenizer-json` flag *after* the model architecture. For example:

```bash
$ python3 scripts/get_tokenizers_json.py
Enter model ID: microsoft/Orca-2-13b
$ mistralrs serve -p 1234 --log output.log -m microsoft/Orca-2-13b --tokenizer-json tokenizer.json
```

Putting it all together, to run, for example, an [Orca](https://huggingface.co/microsoft/Orca-2-13b) model (which does not come with a `tokenizer.json` or chat template):
1) Generate the `tokenizer.json` by running the script at `scripts/get_tokenizers_json.py`. This will output some files including `tokenizer.json` in the working directory.
2) Find and copy the correct chat template from `chat-templates` to the working directory (eg., `cp chat_templates/chatml.json .`)
3) Run `mistralrs serve`, specifying the tokenizer and chat template: `mistralrs serve -p 1234 --log output.txt --chat-template chatml.json -m microsoft/Orca-2-13b -t tokenizer.json`

> Note: For GGUF models, the tokenizer may be loaded directly from the GGUF file by omitting the tokenizer model ID.
