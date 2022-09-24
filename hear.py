def main():
    # ====================
    # Minimise things we do before we start recording
    # ====================
    import threading

    import sounddevice as sd

    initial_seconds = 2
    silence_seconds = 0.5
    max_seconds = 20
    SAMPLE_RATE = 16000

    blocks = []
    block_norms = []
    blocks_per_sec = 5
    cond = threading.Condition()

    def callback(indata, frames, time, status) -> None:
        blocks.append(indata.flatten())  # need to make a copy
        with cond:
            cond.notify()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=SAMPLE_RATE // blocks_per_sec,
        channels=1,
        latency="low",
        callback=callback,
    )
    print("Listening...\a")
    stream.start()
    # ====================

    import math
    import os
    import subprocess

    import numpy as np
    import requests
    import whisper

    assert SAMPLE_RATE == whisper.audio.SAMPLE_RATE

    api_base = os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
    assert api_base.split("/")[-1].startswith("v")
    try:
        api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise RuntimeError("Set the OPENAI_API_KEY environment variable to your OpenAI API key.")
    engine = "text-davinci-002"

    initial_blocks = math.ceil(initial_seconds * blocks_per_sec)
    silence_blocks = math.ceil(silence_seconds * blocks_per_sec)
    max_blocks = math.ceil(max_seconds * blocks_per_sec)
    assert initial_blocks > 0
    assert silence_blocks > 0

    # I couldn't get the model to run on MPS...
    # If you're braver than I am, init the model on CPU, then make it fp32, then move to MPS
    # (and set fp16=False when transcribing). This is enough to make it not crash, but the
    # numerics seem very wrong. First discrepancy is with cross_attn in resblock.
    model = whisper.load_model("base.en")

    while True:
        with cond:
            cond.wait()
        for block in blocks[len(block_norms) :]:
            block_norms.append(np.linalg.norm(block))

        # This seems to work well enough for speech detection
        max_norm = max(block_norms)
        threshold = max(
            min((b for b in block_norms if b > max_norm / 3), default=0),
            np.quantile(block_norms, 2 / initial_blocks),
        )
        if len(blocks) > initial_blocks and all(
            b < threshold for b in block_norms[-silence_blocks:]
        ):
            break
        if len(blocks) > max_blocks:
            break
    stream.stop()
    print("Processing...")
    print()

    audio = np.concatenate(blocks)

    result = model.transcribe(
        audio, prompt="I want to run some shell commands", language="english", fp16=False
    )
    text = result["text"].strip()
    print(text)

    data = {
        "prompt": (
            "Provide a shell script that can be executed in order to "
            "answer the following question:\n"
            f"{text}\n```"
        ),
        "max_tokens": 300,
        "temperature": 0,
        "stop": "```",
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    url = api_base + f"/engines/{engine}/completions"
    resp = requests.post(url, headers=headers, json=data)
    resp.raise_for_status()
    result = resp.json()

    completion = result["choices"][0]["text"].strip()

    print()
    print("Suggestion:\n```")
    print(completion)
    print("```")

    print("Execute? [Y/n] ", end="", flush=True)
    if input().lower() != "n":
        print()
        subprocess.run(["bash", "-c", "set -x\n" + completion])


if __name__ == "__main__":
    main()
