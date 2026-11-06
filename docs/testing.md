# Python quality checks

Every pull request runs `.github/workflows/python-quality.yml` on Ubuntu 24.04
with Python 3.11. The job installs the project in editable mode with its `test`
extra, checks dependency consistency, runs pytest, then runs Black and Ruff in
read-only mode. Formatting and linting still run after a test failure. Pytest's
JUnit XML report is retained as a workflow artifact for 14 days; command output
and the installed dependency versions remain in the Actions logs.

## Reproduce locally

Use a fresh Python 3.11 virtual environment, separate from research environments:

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -r requirements/ci.txt
python -m pip install --no-build-isolation -c requirements/ci.txt -e '.[test]'
python -m pip check
python -m pytest --junitxml=test-results/pytest.xml
python -m black --check src tests scripts app.py
python -m ruff check src tests scripts app.py
```

`requirements/ci.txt` pins direct and transitive dependencies, including pip,
setuptools and wheel. Build isolation is disabled only after these build tools
have been installed, to avoid fetching an unpinned build backend. This lock is
for the CI environment, not a claim that every scientific pipeline has been
validated with these versions. The full existing project dependencies are
installed; no GPU is needed to execute the tests. Linux may install the NCCL
dependency declared by XGBoost, even though the tests do not use it.

Black 24.4.2 and Ruff 0.5.0 match `.pre-commit-config.yaml`. Environment tests
fail if the installed tool versions diverge from the hook revisions. Local
pre-commit hooks may fix files; the CI commands never use `--fix` or format files.
Checks cover Python files in `src/`, `tests/`, `scripts/` and `app.py`, excluding
notebooks, data and archived work. Ruff currently uses its default rule set;
these checks do not constitute a full static or scientific audit.

## Test scope and isolation

Pytest discovers tests only under `tests/`. Strict configuration and markers
catch misspelled settings and markers. The default suite excludes `slow` and
`network` tests, and pytest-socket disables sockets during test execution.
The Gaia regression tests additionally block import-time service access and mock
Gaia/CDS services, retries and sleeps; their catalog files use temporary folders.
They test simulated responses, not live service availability or catalog accuracy.
The environment tests verify socket blocking and shared Black/Ruff versions.

Registered markers are `unit`, `integration`, `slow` and `network`. Future tests
must isolate their data and services; collection-time imports must also avoid
network access. Live service tests require a separate explicit invocation such
as `python -m pytest -o addopts='' -m network` and are outside this CI job.
No private data, credentials, W&B runs or GPU are required by the default suite.
Passing checks establish only the tested contracts, not general scientific
validity. Broader module audits and physical/ML validation are tracked in #43;
the namespace/package redesign remains in #21.

## Updating the lock

Change the tool pins in `requirements/ci.in` together with the hook revisions;
test dependencies live in the `test` extra in `pyproject.toml`. Regenerate the
lock in an isolated environment using uv 0.9.7:

```sh
python -m pip install uv==0.9.7
uv pip compile pyproject.toml requirements/ci.in --extra test --python-version 3.11 --universal --no-annotate --output-file requirements/ci.txt
```

Use `--upgrade` only when intentionally updating resolved versions. Review the
lock diff, recreate a clean environment and run all four checks before sharing.
The lock records versions and platform markers, not artifact hashes or a fixed
Python patch release; the validated hosted target is Ubuntu/Python 3.11.
