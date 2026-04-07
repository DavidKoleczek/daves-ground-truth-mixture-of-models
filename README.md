# Is a Mixture of Models a Competitive Advantage?

Experiments for my blog [Is a Mixture of Models a Competitive Advantage?](https://www.davesgroundtruth.com/is-a-mixture-of-models-a-competitive-advantage/) exploring whether combining multiple frontier AI models to solve tasks outperforms a single model.

## HLE Team Solver Experiment

This experiment compares three different cases of agent(s) trying to solve a random sample of 50 [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) problems.
No agents have web fetch or web search as there is a chance they find the answers.

Baseline:
- Claude Code with Opus 4.6 medium reasoning effort asked to solve the problem directly.

Same Model, Different Personas:
1. Exploration Phase where agents are asked to explore the problem space and write their thoughts and analysis to help a solver in an analysis.md file
    - Claude Code Opus 4.6 medium reasoning effort used with a default system prompt
    - Claude Code Opus 4.6 medium reasoning effort used with a system prompt that encourages more creativity and exploration - "The Contrarian"
    - Claude Code Opus 4.6 medium reasoning effort used with a system prompt that tells it to be a like a top-tier professor - "The Professor"
1. Solver Phase - Claude Code Opus 4.6 medium reasoning effort is given the original question and the analysis.md files from the three explorers, in random order, and asked to solve the problem.

Different Model, Different Personas:
- Same setup as above, but the explorers are different models (Gemini CLI with Gemini-3.1-pro-preview default reasoning and Codex CLI with gpt-5.4 medium reasoning) and randomly assigned the role (default, contrarian, professor):

### Run Experiment

```bash
uv run src/model_diversity/hle/team_solver.py --concurrent 5 --output output/hle
```

### Prerequisites

**API keys** (set as environment variables):
- `ANTHROPIC_API_KEY` -- for Claude Code and grading
- `OPENAI_API_KEY` -- for Codex
- `GEMINI_API_KEY` -- for Gemini CLI

**CLI tools** (must be on PATH):
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) -- `claude` CLI
- [Codex CLI](https://github.com/openai/codex) -- `codex` CLI, started via `codex-app-server-sdk`
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) -- install via `npm install -g @google/gemini-cli`

### Related Work
- [Dynamic Role Assignment for Multi-Agent Debate](https://arxiv.org/pdf/2601.17152)
- [CoMM: Collaborative Multi-Agent, Multi-Reasoning-Path Prompting for Complex Problem Solving](https://arxiv.org/html/2404.17729v1)
- [Town Hall Debate Prompting: Enhancing Logical Reasoning in LLMs through Multi-Persona Interaction](https://arxiv.org/html/2502.15725v1)
- [PersonaFlow: Boosting Research Ideation with LLM-Simulated Expert Personas](https://arxiv.org/html/2409.12538v1)
- [Debate-to-Write: A Persona-Driven Multi-Agent Framework for Diverse Argument Generation](https://aclanthology.org/2025.coling-main.314.pdf)
- [MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework](https://arxiv.org/html/2308.00352v7)
- [ChatDev: Communicative Agents for Software Development](https://arxiv.org/html/2307.07924v5)
- [AgentCoder: Multiagent-Code Generation with Iterative Testing and Optimisation](https://arxiv.org/html/2312.13010v2)
- [Expert Personas Improve LLM Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with PRISM](https://arxiv.org/pdf/2603.18507)


## Infographic Experiment

This experiment explores how infographics can be iterated on to improve them leveraging the visual understanding capabilities of different models.
We use the [nanobanana](https://github.com/gemini-cli-extensions/nanobanana) Gemini CLI extension 
for image generation and then the CLI agents to decide how to iterate on the infographic to improve on it.
The [scenarios file](./src/model_diversity/infographics/scenarios.yaml) contains 10 different scenarios based on 
[infographic-builder](https://github.com/singh2/infographic-builder?ref=davesgroundtruth.com) which is a wonderful project for generating super nice infographics by one of my colleagues. 

Baseline:
- Gemini CLI image that is not iterated on at all, generated with a simple prompt from the [scenarios file](./src/model_diversity/infographics/scenarios.yaml).

Same Model, Different Personas:
1. Generate an infographic with Gemini CLI using a simple prompt from the scenarios file.
1. Claude Code is asked to evaluate and determine if the infographic should be improved, in this order:
    - Information Architecture and Content Accuracy
    - Visual Communication Effectiveness
    - Visual Design and Aesthetic Cohesion
1. The final image is saved for comparison

Different Model, Different Personas:
1. Same setup as above, but the model used to evaluate and determine changes is randomly assigned between Claude Code, Codex CLI, and Gemini CLI (Gemini CLI with Gemini-3.1-pro-preview default reasoning and Codex CLI with gpt-5.4 medium reasoning).

### Run Experiment

```bash
uv run src/model_diversity/infographics/experiment.py --concurrent 3 --output output/infographics
```

### Prerequisites

**API keys** (set as environment variables):
- `ANTHROPIC_API_KEY` -- for Claude Code
- `OPENAI_API_KEY` -- for Codex
- `GEMINI_API_KEY` -- for Gemini CLI and nanobanana image generation

**CLI tools** (must be on PATH):
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) -- `claude` CLI
- [Codex CLI](https://github.com/openai/codex) -- `codex` CLI, started via `codex-app-server-sdk`
- [Gemini CLI](https://github.com/google-gemini/gemini-cli) -- install via `npm install -g @google/gemini-cli`

**Gemini CLI extensions**:
- [nanobanana](https://github.com/gemini-cli-extensions/nanobanana) -- install via `gemini extensions install https://github.com/gemini-cli-extensions/nanobanana`
  - During installation, provide your `GEMINI_API_KEY` when prompted for the API key
  - If the interactive installer does not work, clone the repo into `~/.gemini/extensions/nanobanana/`, run `npm install` inside `mcp-server/`, and add `"NANOBANANA_API_KEY": "${GEMINI_API_KEY}"` to the `env` field in `gemini-extension.json`
