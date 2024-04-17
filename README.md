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


## CLI Usage

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

To create 2D animations of the embeddings, use the CLI command `clip-eval animate`.
This command allows to visualise the reduction of embeddings from two models on the same dataset.

The animations will be saved at the location specified by the environment variable `CLIP_EVAL_OUTPUT_PATH`.
By default, this path corresponds to the repository directory.


## Datasets

This repository contains classification datasets sourced from [Hugging Face](https://Hugging Face.co/datasets) and [Encord](https://app.encord.com/projects).
> Currently, only image and image groups datasets are supported, with potential for future expansion to include video datasets.

| Dataset Title             | Source       | Title in Source                      |
|:--------------------------|:-------------|:-------------------------------------|
| Alzheimer-MRI             | Hugging Face | Falah/Alzheimer_MRI                  |
| chest-xray-classification | Hugging Face | trpakov/chest-xray-classification    |
| LungCancer4Types          | Hugging Face | Kabil007/LungCancer4Types            |
| plants                    | Hugging Face | sampath017/plants                    |
| rsicd                     | Encord       | -                                    |
| skin-cancer               | Hugging Face | marmal88/skin_cancer                 |
| sports-classification     | Hugging Face | HES-XPLAIN/SportsImageClassification |

### Add a Dataset from a Known Source

To register a dataset from a known source, you can include the dataset definition as a JSON file in the `sources/datasets` folder.
The definition will be validated against the schema defined by the `clip_eval.dataset.base.DatasetDefinitionSpec` Pydantic class to ensure that it adheres to the required structure.
You can find the explicit schema in `sources/dataset-definition-schema.json`.

Check out the declarations of known sources at `clip_eval.dataset.types` and refer to the existing dataset definitions in the `sources/datasets` folder for guidance.
Below is an example of a dataset definition for the [plants](https://huggingface.co/datasets/sampath017/plants) dataset sourced from Hugging Face:
```json
{
  "dataset_type": "HFDataset",
  "title": "plants",
  "title_in_source": "sampath017/plants"
}
```

In each dataset definition, the `dataset_type` and `title` fields are required.
The `dataset_type` indicates the name of the class that represents the source, while `title` serves as a reference for the dataset on this platform.
For Hugging Face datasets, the `title_in_source` field should store the title of the dataset as it appears on the Hugging Face website.

For datasets sourced from Encord, other set of fields are required. These include `project_hash`, which contains the hash of the project, and `classification_hash`, which contains the hash of the radio-button (multiclass) classification used in the labels.  

### Programmatically Add a Dataset

Alternatively, you can programmatically add a dataset, which will be available only for the current session, using the `register_dataset()` method of the `clip_eval.dataset.DatasetProvider` class. 

Here is an example of how to register a dataset from Hugging Face using Python code:
```python
from clip_eval.dataset import DatasetProvider, Split
from clip_eval.dataset.types import HFDataset

DatasetProvider.register_dataset(HFDataset, "plants", title_in_source="sampath017/plants")
ds = DatasetProvider.get_dataset("plants", split=Split.ALL)
print(len(ds))  # Returns: 219
```

### Remove a dataset

To permanently remove a dataset, simply delete the corresponding JSON file stores in the `sources/datasets` folder.
This action removes the dataset from the list of available datasets in the CLI, disabling the option to create any further embedding using its data.
However, all embeddings previously built on that dataset will remain intact and available for other tasks such as evaluation and animation.

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
