# mistralrs PyO3 Bindings

To use, activate a Python virtual enviornment and ensure that `maturin` is installed, for example

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install maturin
```

And then install `mistralrs` by executing the following.

```bash
maturin develop -r --features ...
```

Features such as `cuda` or `flash-attn` may be specified with the `--features` argument.