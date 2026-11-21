#!/bin/sh
# Fetch the Du et al. (2023) debate code at the pinned commit into upstream/ and
# check the SHA-256 of every file used (see README.md). Needs git and network access.
set -e
cd "$(dirname "$0")"
COMMIT=9846749350eb917ae5bfaaff4c645fc705b8d3af
TMP=$(mktemp -d)
git clone --quiet https://github.com/composable-models/llm_multiagent_debate "$TMP/repo"
git -C "$TMP/repo" checkout --quiet "$COMMIT"
mkdir -p upstream/gsm upstream/math upstream/mmlu
for f in README.md requirements.txt gsm/gen_gsm.py gsm/eval_gsm.py math/gen_math.py mmlu/gen_mmlu.py mmlu/eval_mmlu.py; do
  cp "$TMP/repo/$f" "upstream/$f"
done
rm -rf "$TMP"
cd upstream
cat > ../SHA256SUMS <<'SUMS'
7fcae591f15a0d14545d1fcbcfe41c3c090a92305a01e13064f443b4d1b04733  README.md
7f37c317e9a0fe5636423b099f11355eca899c69a657981610a28fcc5b3c673b  requirements.txt
3a4adafb7a2563e70eb86855aad4d99507f12f8382f39dda2af29baa4bf1900a  gsm/gen_gsm.py
dd500b00b7be92a7cc584a2190349704dfe5416abc3079ddf2aae56ba38ff1e6  gsm/eval_gsm.py
cbaa63faa2cb7b0bcb7afb954fd3cb1c31fc37c6a3d927c34c51a48c13a2836e  math/gen_math.py
23ffcf0b0f78da2bb99744946844f93477371fc5eaaa4ce1969fa6e42b9cf3c2  mmlu/gen_mmlu.py
0987d705729c591467f06eacbf32db9038ed0c7c402348181641e14994b73018  mmlu/eval_mmlu.py
SUMS
if command -v sha256sum >/dev/null 2>&1; then sha256sum -c ../SHA256SUMS; else shasum -a 256 -c ../SHA256SUMS; fi
rm ../SHA256SUMS
echo "ok: upstream files fetched and verified"
