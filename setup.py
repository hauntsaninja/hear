from setuptools import setup

setup(
    name="hear",
    version="0.1.0",
    py_modules=["hear"],
    entry_points={"console_scripts": ["hear=hear:main"]},
    install_requires=[
        "numpy",
        "requests",
        "sounddevice",
        "whisper @ git+https://github.com/openai/whisper.git",
    ],
)
