# docker run --rm -v .:/io ghcr.io/pyo3/maturin build --release -o wheels -m mistralrs-pyo3/Cargo.toml

docker build -t wheelmaker:latest -f Dockerfile.manylinux .

# Manylinux and OSX

docker run --rm -v .:/io wheelmaker build --release -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.10
docker run --rm -v .:/io wheelmaker build --release -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.11
docker run --rm -v .:/io wheelmaker build --release -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.12

maturin build -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.10
maturin build -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.11
maturin build -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.12

# Metal

maturin build -o wheels-metal -m mistralrs-pyo3/Cargo.toml --interpreter python3.10 --features metal
maturin build -o wheels-metal -m mistralrs-pyo3/Cargo.toml --interpreter python3.11 --features metal
maturin build -o wheels-metal -m mistralrs-pyo3/Cargo.toml --interpreter python3.12 --features metal

# Accelerate

maturin build -o wheels-accelerate -m mistralrs-pyo3/Cargo.toml --interpreter python3.10 --features accelerate
maturin build -o wheels-accelerate -m mistralrs-pyo3/Cargo.toml --interpreter python3.11 --features accelerate
maturin build -o wheels-accelerate -m mistralrs-pyo3/Cargo.toml --interpreter python3.12 --features accelerate
