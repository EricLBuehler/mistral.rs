# MAC: Aarch64 Manylinux and OSX

docker build -t wheelmaker:latest -f Dockerfile.manylinux .
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

# WINDOWS: x86_64 Manylinux, Windows

docker build -t wheelmaker:latest -f Dockerfile.manylinux .
docker run --rm -v .:/io wheelmaker build --release -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.10
docker run --rm -v .:/io wheelmaker build --release -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.11
docker run --rm -v .:/io wheelmaker build --release -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.12

maturin build -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.10
maturin build -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.11
maturin build -o wheels-cpu -m mistralrs-pyo3/Cargo.toml --interpreter python3.12

# CUDA

maturin build -o wheels-cuda -m mistralrs-pyo3/Cargo.toml --interpreter python3.10 --features cuda
maturin build -o wheels-cuda -m mistralrs-pyo3/Cargo.toml --interpreter python3.11 --features cuda
maturin build -o wheels-cuda -m mistralrs-pyo3/Cargo.toml --interpreter python3.12 --features cuda

# MKL

maturin build -o wheels-mkl -m mistralrs-pyo3/Cargo.toml --interpreter python3.10 --features mkl
maturin build -o wheels-mkl -m mistralrs-pyo3/Cargo.toml --interpreter python3.11 --features mkl
maturin build -o wheels-mkl -m mistralrs-pyo3/Cargo.toml --interpreter python3.12 --features mkl

### UPLOADING

# ⚠️⚠️⚠️⚠️ Be sure to update the `project.name` field in `pyproject.toml`!! ⚠️⚠️⚠️⚠️
# mistralrs, mistralrs-cuda, mistralrs-metal, mistralrs-mkl, mistralrs-accelerate

## testpypi:
# twine upload --repository pypi --password PASSWORD --username __token__ wheels-NAME/*.whl


## pypi:
# twine upload --repository pypi --password PASSWORD --username __token__ wheels-cuda/*.whl
# twine upload --repository pypi --password PASSWORD --username __token__ wheels-mkl/*.whl
# twine upload --repository pypi --password PASSWORD --username __token__ wheels-cuda/*.whl
# twine upload --repository pypi --password PASSWORD --username __token__ wheels-metal/*.whl
# ⚠️ Need both x86_64 and aarch64 builds before this! ⚠️
# twine upload --repository pypi --password PASSWORD --username __token__ wheels-cpu/*.whl

