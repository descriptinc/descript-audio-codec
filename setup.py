from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="descript-audio-codec",
    version="1.0.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Software Development :: Libraries",
    ],
    description="A high-quality general neural audio codec.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Prem Seetharaman, Rithesh Kumar",
    author_email="prem@descript.com",
    url="https://github.com/descriptinc/descript-audio-codec",
    license="MIT",
    packages=find_packages(),
    keywords=["audio", "compression", "machine learning"],
    install_requires=[
        "argbind>=0.3.7",
        "descript-audiotools>=0.7.2",
        "einops",
        "numpy",
        "soundfile==0.13.1",
        "torch",
        "torchaudio",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "pynvml",
            "psutil",
            "pandas",
            "onnx",
            "onnx-simplifier",
            "seaborn",
            "jupyterlab",
            "pandas",
            "watchdog",
            "pesq",
            "tabulate",
            "encodec",
            "wandb",
            "speechbrain>=0.5.0",  # For SIM (Speaker Identification Metric)
            "openai-whisper>=20231117",  # For WER (Word Error Rate)
            "jiwer>=3.0.0",  # For WER calculation
        ],
    },
)
