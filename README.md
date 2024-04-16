# clip-eval

Welcome to `clip-eval`, a repository for evaluating text-to-image models like CLIP, SigLIP, and the like.

Evaluate machine learning models against a benchmark of datasets to assess their performance on the generated embeddings, and visualize changes in embeddings from one model to another within the same dataset.

## Installation

> `clip-eval` requires [Python 3.11](https://www.python.org/downloads/release/python-3115/) and [Poetry](https://python-poetry.org/docs/#installation).

1. Clone the repository:
   ```
   git clone https://github.com/encord-team/text-to-image-eval.git
   ```
2. Navigate to the project directory:
   ```
   cd text-to-image-eval
   ```
3. Install the required dependencies:
   ```
   poetry shell
   poetry install
   ```
4. Add environment variables:
   ```
   export CLIP_EVAL_CACHE_PATH=$PWD/.cache
   export CLIP_EVAL_OUTPUT_PATH=$PWD/output
   export ENCORD_SSH_KEY_PATH=<path_to_the_encord_ssh_key_file>
   ```


## Usage

### Embeddings generation

To build embeddings, run the CLI command `clip-eval build`.
This commands allows you to interactively select the model and dataset combinations on which to build the embeddings.

Alternatively, you can choose known (model, dataset) pairs using the `--model-dataset` option. For example:
```
clip-eval build --model-dataset clip/plants
```

### Model evaluation

To evaluate models, use the CLI command `clip-eval evaluate`.
This command enables interactive selection of model and dataset combinations for evaluation.

Alternatively, you can specify known (model, dataset) pairs using the `--model-dataset` option. For example:
```
clip-eval evaluate --model-dataset clip/plants
```

### Embeddings animation

To create a 2D animation of the embeddings, use the CLI command `clip-eval animate`.
This command allows to visualise the reduction of embeddings from two different models on the same dataset.

The animations will be saved at the location specified by the environment variable `CLIP_EVAL_OUTPUT_PATH`.
By default, this path corresponds to the repository directory.


## Set up the development environment

1. Create the virtual environment, add dev dependencies and set up pre-commit hooks.
   ```
   ./dev-setup.sh
   ```
2. Add environment variables:
   ```
   export CLIP_EVAL_CACHE_PATH=$PWD/.cache
   export CLIP_EVAL_OUTPUT_PATH=$PWD/output
   export ENCORD_SSH_KEY_PATH=<path_to_the_encord_ssh_key_file>
   ```


## Contributing
Contributions are welcome!
Please feel free to open an issue or submit a pull request with your suggestions, bug fixes, or new features.
