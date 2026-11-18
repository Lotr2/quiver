# Contributing

## Build and test

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Useful options:

| Option | Effect |
|---|---|
| `-DQUIVER_SANITIZE=address,undefined` | AddressSanitizer + UBSan |
| `-DQUIVER_SANITIZE=thread` | ThreadSanitizer (run `--gtest_filter='Hnsw.*:Collection.*'`) |
| `-DQUIVER_NATIVE=ON` | `-march=native` for local benchmarking |
| `-DQUIVER_WARNINGS_AS_ERRORS=ON` | what CI uses on Linux |

Format C++ with `clang-format -i` (style in `.clang-format`); CI checks it with clang-format 21.

## Commit messages

This repository follows [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <imperative summary, lower case, no period>

<body: what changed and why, wrapped at 72 columns>
```

| Type | Use for |
|---|---|
| `feat` | a new capability |
| `fix` | a bug fix |
| `perf` | a change that only makes things faster |
| `refactor` | restructuring with no behaviour change |
| `test` | adding or fixing tests |
| `docs` | documentation only |
| `build` / `ci` | build system, dependencies, pipelines |
| `style` | formatting only |
| `chore` | anything else |

Scopes in use: `core`, `index`, `storage`, `query`, `server`, `bench`, `client`.

Keep each commit to one logical change that builds and passes tests on its own. The body should explain *why*, because the diff already shows *what*.

## Pull requests

- Add a test with every bug fix that fails without the fix.
- For performance changes, include `quiver-bench` output from before and after on the same machine.
- Changes to the on-disk format must bump its version number and keep the loader rejecting files it does not understand.
