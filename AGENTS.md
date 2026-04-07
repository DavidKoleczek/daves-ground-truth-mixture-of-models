# General Instructions
- This is a Python workspace for blog analysis and research. Follow best Python practices.
- Make sure any comments in code are necessary. Comments should capture the "why", not the "what".
- Do not run tests automatically unless asked.
- No emojis or em dashes in writing.
- Always use the most up to date frontier models as of April 2, 2026 this is:
    - Anthropic: `claude-opus-4.6`
    - OpenAI: `gpt-5.4`
    - Google Gemini: `gemini-3.1-pro-preview`
- The `ai_working/` directory at the root will contain cloned repos we are referencing for easier exploration.

# Python Development Instructions
- `ty` by Astral is used for type checking. Always add appropriate type hints.
- Follow the Google Python Style Guide.
- After each code change, checks are automatically run. Fix any issues that arise.
- **IMPORTANT**: The checks will remove any unused imports after you make an edit to a file. So if you need to use a new import, be sure to use it FIRST (or do your edits at the same time) or else it will be automatically removed. DO NOT use local imports to get around this.
- Always prefer pathlib for dealing with files. Use `Path.open` instead of `open`.
- When using pathlib, **always** use `.parents[i]` syntax to go up directories instead of using `.parent` multiple times.
- When writing tests, use pytest and pytest-asyncio.
- NEVER use `# type: ignore`. It is better to leave the issue and have the user work with you to fix it.
- `__init__.py` files should always be left empty unless absolutely necessary.
- The tests can take a while since they often test agents end to end, you should run them individually as needed.
- ALWAYS keep constants and imports at the top of files.

# Key Files

@README.md

@pyproject.toml
