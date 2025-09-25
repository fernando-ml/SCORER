# SCORER: Stackelberg Coupling of Online Representation  Learning and Reinforcement Learning
Official implementation of the SCORER framework.

This code was developed with Python 3.10. The project uses `uv` for dependency management.

If you don't have `uv` installed, you can install it following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

To set up the project:
```bash
# Install dependencies and create virtual environment
uv sync
```

The project will automatically use Python 3.10 as specified in the `.python-version` file.

## Running

Use `uv run` to execute scripts, for example:

```bash
uv run DDQN_SCORER.py --env SpaceInvaders-MinAtar --total_timesteps 1e8 --use_be_variance --seed 10 --num_seeds 30
```
