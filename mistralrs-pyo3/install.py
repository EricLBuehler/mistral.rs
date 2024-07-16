import requests
import subprocess
import os

features = {
    "cuda": "mistralrs-cuda",
    "accelerate": "mistralrs-accelerate",
    "mkl": "mistralrs-mkl",
    "metal": "mistralrs-metal",
    "none": "mistralrs",
}

def install_rust():
    # Make the HTTPS request
    response = requests.get('https://sh.rustup.rs')

    # Check if the request was successful
    if response.status_code == 200:
        # Run the shell script
        subprocess.run(['sh', '-s', '--', '-y'], input=response.text, text=True)
        env = os.environ.copy()

        # Add the directory containing cargo to the PATH
        env['PATH'] += os.pathsep + os.path.expanduser("$HOME/.cargo/env")
        # Source cargo
        result = subprocess.run(['. "$HOME/.cargo/env"'], shell=True, env=env)
        if result.returncode != 0:
            print("Error occurred:")
            print("Standard Output:", result.stdout.decode())
            print("Standard Error:", result.stderr.decode())
        else:
            print("Success")
            current_path = os.environ.get('PATH', '')

            # Append the Rust bin directory to the PATH
            rust_bin_path = os.path.expanduser("~/.cargo/bin")
            new_path = f"{rust_bin_path}:{current_path}"

            # Update the PATH environment variable
            os.environ['PATH'] = new_path
            
    else:
        print("Failed to retrieve the script.")

def install_mistralrs(feature: str):
    global features

    env = os.environ.copy()
    env["CUDA_NVCC_FLAGS"] = "-fPIE"
    print(f"Installing mistral.rs: {features[feature]}")
    result = subprocess.run(['pip', 'install', features[feature], '-v'], env=env)
    print(result)

feature = input(f"Enter feature, one of {list(features.keys())} or 'none' for none: ")

if feature not in features:
    raise ValueError(f"Feature not in features, expected one of {features.keys()} but got {feature}")

install_rust()
install_mistralrs(feature)