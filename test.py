import requests
data = {"prompt": "The best programming language is Rust because", 
        "sampling_params": {"temperature": None, "top_k": None, "top_p": None, "top_n_logprobs": 1, "repeat_penalty": 1.1, "stop_toks": None, "max_len": 1024}}
r = requests.get("http://localhost:1234/", json=data)
print(r)