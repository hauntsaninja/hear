# hear

A voice assistant for the command line.

Uses OpenAI's [whisper](https://github.com/openai/whisper) for (on-device) speech to text.

Uses OpenAI's [API](https://openai.com/api/) to generate shell commands based on your spoken question.

## Installation

```
pip install git+https://github.com/hauntsaninja/hear.git
```
Requires `OPENAI_API_KEY` environment variable to be set.

If you have trouble installing, see the [whisper](https://github.com/openai/whisper)
installation instructions.

## Example

````
λ hear
Listening...
Processing...

I just added a readme, could you commit it?

Suggestion:
```
git add README.md
git commit -m "added a readme"
```
Execute? [Y/n]

+ git add README.md
+ git commit -m 'added a readme'
[main 3477818] added a readme
 1 file changed, 7 insertions(+)
 create mode 100644 README.md
 ````
