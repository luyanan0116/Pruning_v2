# v7 validation

Static/runtime checks performed in the build environment:

```text
python -m compileall -q .                  PASS
PYTHONPATH=. pytest -q                     24 passed
bash -n scripts/*.sh                       PASS
synthetic unit-local GB + 3-repeat LCB     PASS; non-zero empirical std
```

The synthetic LCB smoke test produced a non-zero mean empirical standard deviation, confirming that the repeated path is active rather than the v6 `std=0` fallback.

A full Llama-2/C4/WikiText PPL run was not executed in this build environment because the model/data are not present and the environment does not currently have `transformers` installed. `requirements.txt` already declares the required runtime dependency. No fixed PPL value is hard-coded or claimed by this package.
