"""
Assemble the Hugging Face Space (Gradio demo) into one folder, ready to upload:

    python tools/build_space.py
    hf upload Efe-Tatar/base-100m-demo build/space . --repo-type=space

The Space needs the demo (space/app.py) plus the parts of this project it runs on:
the model and tokenizer code and the Hub loader. Nothing else (no training code, no data).
"""
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILES = [
	"config.py",
	"network/__init__.py", "network/transformer.py", "network/causal_lm.py",
	"tokenizer/bpe.py",
	"tools/__init__.py", "tools/export_hf.py", "tools/load_hf.py",
	"space/app.py",
]
# the Space reads these two from its root
ROOT_FILES = {"space/README.md": "README.md", "space/requirements.txt": "requirements.txt"}


def main(out=os.path.join(ROOT, "build", "space")):
	shutil.rmtree(out, ignore_errors=True)
	for rel in FILES:
		os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
		shutil.copy(os.path.join(ROOT, rel), os.path.join(out, rel))
	for src, dst in ROOT_FILES.items():
		shutil.copy(os.path.join(ROOT, src), os.path.join(out, dst))
	print(f"Space assembled in {out}:")
	for dirpath, _, files in os.walk(out):
		for f in sorted(files):
			print("  " + os.path.relpath(os.path.join(dirpath, f), out))


if __name__ == "__main__":
	main()
