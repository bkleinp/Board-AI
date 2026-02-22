# Board AI

Board AI is a local voice-controlled visual workspace.

It listens to your microphone, streams live transcription in the browser, and converts spoken commands into HTML visuals on the right side of the screen.

## Requirements

- Python 3.10+
- A working microphone
- An OpenAI API key

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run

```bash
python main.py
```

Then open [http://localhost:8000](http://localhost:8000).

## Basic Use

1. Click **Start Transcription**.
2. Speak normally to see live transcript updates.
3. Speak commands using the pattern: `orange ... please`.

Example: `orange make the screen split red and blue please`
