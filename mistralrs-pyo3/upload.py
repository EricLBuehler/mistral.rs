#! /usr/bin/python3
# chmod +x upload.py
import os
import time
from string import Template

with open("Cargo_template.toml", "r") as cargo_file:
    data = cargo_file.read()
    assert 'features=["$feature_name"]' in data
    cargo_template = Template(data)

with open("pyproject_template.toml", "r") as pyproject_file:
    data = pyproject_file.read()
    assert 'name = "$name"' in data
    pyproject_template = Template(data)


with open("Cargo.toml", "r") as cargo_file:
    cargo_original = cargo_file.read()

with open("pyproject.toml", "r") as pyproject_file:
    pyproject_original = pyproject_file.read()

if os.path.exists("dist"):
    print("`dist/*` exists. Please remove it before uploading.")

features = {
    "cuda": "mistralrs-cuda",
    "accelerate": "mistralrs-accelerate",
    "mkl": "mistralrs-mkl",
    "metal": "mistralrs-metal",
}

try:
    for feature, name in features.items():
        cargo_toml = cargo_template.safe_substitute(feature_name=feature)
        pyproject_toml = pyproject_template.safe_substitute(name=name)
        with open("Cargo.toml", "w") as cargo_file:
            cargo_file.write(cargo_toml)
        with open("pyproject.toml", "w") as pyproject_file:
            pyproject_file.write(pyproject_toml)

        os.system("maturin sdist --out dist")
finally:
    with open("Cargo.toml", "w") as cargo_file:
        cargo_file.write(cargo_original)
    with open("pyproject.toml", "w") as pyproject_file:
        pyproject_file.write(pyproject_original)


print(
    f"⭐ Generated sdists for features {[k for k,_ in features.items()]} with corresponding names {[v for _,v in features.items()]}"
)

# Generate CPU lib
os.system("maturin sdist --out dist")

if "PYPI_TOKEN" in os.environ:
    password = os.environ["PYPI_TOKEN"]
    print("Using PyPi token as environment variable")
    target = "pypi"
else:
    password = input("Enter API token: ")
    target = input("Enter target (testpypi or pypi): ")
print(f"🚀 Uploading to {target} in 5 seconds. Press <CTRL>-C to abort.")
time.sleep(5)

# Upload
os.system(
    f"twine upload --repository {target} --password {password} --username __token__ dist/*"
)

# Clear tmp files
os.system("rm -rf dist")
