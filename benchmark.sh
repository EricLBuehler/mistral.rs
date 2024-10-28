curl https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf -L -o Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf
curl https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -L -o Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
python3 -m mlx_lm.convert --hf-path meta-llama/Meta-Llama-3.1-8B-Instruct -q  --q-bits 4 --mlx-path llama3.1_8b_4bit
python3 -m mlx_lm.convert --hf-path meta-llama/Meta-Llama-3.1-8B-Instruct -q  --q-bits 8 --mlx-path llama3.1_8b_8bit

cargo run --features metal --release --bin mistralrs-bench '--' --n-gen 256 --n-prompt 0 --isq q4k plain -m meta-llama/Llama-3.1-8B-Instruct
./llama-cli -m ../gguf_models/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf -n 256
python3 -m mlx_lm.generate --model mlx_models/llama3.1_8b_4bit --prompt "What is an LLM?" -m 256

cargo run --features metal --release --bin mistralrs-bench '--' --n-gen 256 --n-prompt 0 --isq q8_0 plain -m meta-llama/Llama-3.1-8B-Instruct
./llama-cli -m ../gguf_models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -n 256
python3 -m mlx_lm.generate --model mlx_models/llama3.1_8b_8bit --prompt "What is an LLM?" -m 256



curl https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_L.gguf -L -o Phi-3.5-mini-instruct-Q4_K_L.gguf
curl https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q8_0.gguf -L -o Phi-3.5-mini-instruct-Q8_0.gguf

cargo run --features metal --release --bin mistralrs-bench '--' --n-gen 256 --n-prompt 0 --isq q4k plain -m microsoft/Phi-3.5-mini-instruct
./llama-cli -m ../gguf_models/Phi-3.5-mini-instruct-Q4_K_L.gguf -n 256

cargo run --features metal --release --bin mistralrs-bench '--' --n-gen 256 --n-prompt 0 --isq q8_0 plain -m microsoft/Phi-3.5-mini-instruct
./llama-cli -m ../gguf_models/Phi-3.5-mini-instruct-Q8_0.gguf -n 256

