# Research Project Starter Repository

This repository is a starting point for Michael's usual layout for data
experiments.

## Using the Template

You can drop these files into a new project to start with a reasonable starting
point for a data analytics project, especially using Python.

The most important files to edit are this file, to add instructions and
description of your particular project, and `pyproject.toml` to add your
software dependencies.  You may also want to use Pixi to manage dependencies
instead of `uv`.

## Software Setup

This template is designed to work with the following tools:

-   Visual Studio Code for editing.  Other editors work fine too, but it
    provides settings files to get started quickly with VSCode.
-   [`uv`][uv] for managing Python environments and dependencies.  Run `uv sync`
    to set up an environment with the project dependencies.  You can install UV
    itself from the web site, Homebrew (`brew install uv`), or in Windows using
    WinGet (`winget install astral-sh.uv`).
-   `pre-commit` for enforcing source code formatting and standards.
    `pre-commit` is included in the development dependencies, so it will be
    installed in your virtual environment when you run `uv sync`; you can also
    install it on your system to be able to run the pre-commit hooks without the
    software environment.

[uv]: https://astral.sh/uv/

### Installation

1.  Install `uv` and, optionally, `pre-commit`.
    -   Mac: `brew install uv pre-commit`
    -   Windows: `winget install astral-sh.uv`
    -   Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2.  Install project dependencies with `uv sync`.  This will create a virtual
    environment in `.venv`, which you can activate in your shell:

    ```console
    $ uv sync
    $ . ./.venv/bin/activate
    ```

3.  Set up `pre-commit`:
    ```console
    $ pre-commit install
    ```

## Directory Layout

My usual layout is like this:

-   `data/`: contains the project's *input* data, usually tracked with DVC.

-   `src/`: contains the source code specific to this project.  There is usually
    a Python package in this directory, e.g. `myproject`, to house the code and
    make it easier to input.

    Sometimes the scripts live under `src`, either in the project package or in
    a separate `scripts` directory; other times, however, they live in the
    directories in which they do their work.  Whatever is clearest for a
    particular project.

    With the package in `src/`, and automatically set up into the virtual
    environment with `uv sync`, its contents can be imported by scripts and
    notebooks throughout the project repository.

-   Other directories to contain different classes of outputs.  Sometimes this
    is organized by data set; for a project with a single data set, it is often
    organized by stage, such as `models/` and `recommendations/`.
