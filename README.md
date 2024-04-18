# clip-eval

Welcome to `clip-eval`, a repository for benchmarking text-to-image models **on your own data**!

> Evaluate your (or HF) text-to-image embedding models like [CLIP][openai/clip-vit-large-patch14-336] from OpenAI against your (or HF) datasets to estimate how well the model will perform on your classification dataset.

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

### Embeddings Generation

To build embeddings, run the CLI command `clip-eval build`.
This commands allows you to interactively select the model and dataset combinations on which to build the embeddings.

Alternatively, you can choose known (model, dataset) pairs using the `--model-dataset` option. For example:

```
clip-eval build --model-dataset clip/plants
```

### Model Evaluation

To evaluate models, use the CLI command `clip-eval evaluate`.
This command enables interactive selection of model and dataset combinations for evaluation.

Alternatively, you can specify known (model, dataset) pairs using the `--model-dataset` option. For example:

```
clip-eval evaluate --model-dataset clip/plants
```

### Embeddings Animation

To create 2D animations of the embeddings, use the CLI command `clip-eval animate`.
This command allows to visualise the reduction of embeddings from two models on the same dataset.

The animations will be saved at the location specified by the environment variable `CLIP_EVAL_OUTPUT_PATH`.
By default, this path corresponds to the repository directory.

## Datasets

This repository contains classification datasets sourced from [Hugging Face](https://huggingface.co/datasets) and [Encord](https://app.encord.com/projects).

> ⚠️ Currently, only image and image groups datasets are supported, with potential for future expansion to include video datasets.

| Dataset Title             | Implementation                  | HF Dataset                                                                           |
| :------------------------ | :------------------------------ | :----------------------------------------------------------------------------------- |
| Alzheimer-MRI             | [Hugging Face][hf-dataset-impl] | [Falah/Alzheimer_MRI][Falah/Alzheimer_MRI]                                           |
| chest-xray-classification | [Hugging Face][hf-dataset-impl] | [trpakov/chest-xray-classification][trpakov/chest-xray-classification]               |
| LungCancer4Types          | [Hugging Face][hf-dataset-impl] | [Kabil007/LungCancer4Types][Kabil007/LungCancer4Types]                               |
| plants                    | [Hugging Face][hf-dataset-impl] | [sampath017/plants][sampath017/plants]                                               |
| skin-cancer               | [Hugging Face][hf-dataset-impl] | [marmal88/skin_cancer][marmal88/skin_cancer]                                         |
| sports-classification     | [Hugging Face][hf-dataset-impl] | [HES-XPLAIN/SportsImageClassification][HES-XPLAIN/SportsImageClassification]         |
| rsicd                     | [Encord][encord-dataset-impl]   | <span style="color: red">\*</span> Requires ssh key and access to the Encord project |

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

### Add a Dataset Source

Expanding the dataset sources involves two key steps:

1. Create a dataset class that inherits from `clip_eval.dataset.Dataset` and specifies the input requirements for extracting data from the new source.
   This class should encapsulate the necessary logic for fetching and processing dataset elements.
2. Generate a dataset definition in JSON format and save it in the `sources/datasets` folder, following the guidelines outlined in the previous section.
   Ensure that the definition includes essential fields such as `dataset_type`, `title`, and `module_path`, which points to the file containing the dataset class implementation.

> It's recommended to store the file containing the dataset class implementation in the `clip_eval/dataset/types` folder and add a reference to the class in the `__init__.py` file in the same folder.
> This ensures that the new dataset type is accessible by default for all dataset definitions, eliminating the need to explicitly state the `module_path` field for datasets from such source.

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

### Remove a Dataset

To permanently remove a dataset, simply delete the corresponding JSON file stores in the `sources/datasets` folder.
This action removes the dataset from the list of available datasets in the CLI, disabling the option to create any further embedding using its data.
However, all embeddings previously built on that dataset will remain intact and available for other tasks such as evaluation and animation.

## Models

This repository contains models sourced from [Hugging Face](https://huggingface.co/models), [OpenCLIP](https://github.com/mlfoundations/open_clip) and local implementations based on OpenCLIP models.

_TODO_: Some more prose about what's the difference between implementations.

### Hugging Face Models

| Model Title      | Implementation                | HF Model                                                                                       |
| :--------------- | :---------------------------- | :--------------------------------------------------------------------------------------------- |
| apple            | [OpenCLIP][open-model-impl]   | [apple/DFN5B-CLIP-ViT-H-14][apple/DFN5B-CLIP-ViT-H-14]                                         |
| apple            | [OpenCLIP][open-model-impl]   | [apple/DFN5B-CLIP-ViT-H-14][apple/DFN5B-CLIP-ViT-H-14]                                         |
| bioclip          | [OpenCLIP][open-model-impl]   | [imageomics/bioclip][imageomics/bioclip]                                                       |
| eva-clip         | [OpenCLIP][open-model-impl]   | [BAAI/EVA-CLIP-8B-448][BAAI/EVA-CLIP-8B-448]                                                   |
| vit-b-32-laion2b | [OpenCLIP][local-model-impl]  | [laion/CLIP-ViT-B-32-laion2B-s34B-b79K][ViT-B-32]                                              |
| clip             | [Hugging Face][hf-model-impl] | [openai/clip-vit-large-patch14-336][openai/clip-vit-large-patch14-336]                         |
| fashion          | [Hugging Face][hf-model-impl] | [patrickjohncyh/fashion-clip][patrickjohncyh/fashion-clip]                                     |
| plip             | [Hugging Face][hf-model-impl] | [vinid/plip][vinid/plip]                                                                       |
| pubmed           | [Hugging Face][hf-model-impl] | [flaviagiammarino/pubmed-clip-vit-base-patch32][flaviagiammarino/pubmed-clip-vit-base-patch32] |
| rsicd            | [Hugging Face][hf-model-impl] | [flax-community/clip-rsicd][flax-community/clip-rsicd]                                         |
| siglip_large     | [Hugging Face][hf-model-impl] | [google/siglip-large-patch16-256][google/siglip-large-patch16-256]                             |
| siglip_small     | [Hugging Face][hf-model-impl] | [google/siglip-base-patch16-224][google/siglip-base-patch16-224]                               |
| street           | [Hugging Face][hf-model-impl] | [geolocal/StreetCLIP][geolocal/StreetCLIP]                                                     |
| tinyclip         | [Hugging Face][hf-model-impl] | [wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M][wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M]     |

### Locally Trained Models

| Model Title  | Implementation                    | Weights |
| :----------- | :-------------------------------- | :------ |
| rsicd-encord | [LocalOpenCLIP][local-model-impl] | -       |

### Add a Model from a Known Source

To register a model from a known source, you can include the model definition as a JSON file in the `sources/models` folder.
The definition will be validated against the schema defined by the `clip_eval.model.base.ModelDefinitionSpec` Pydantic class to ensure that it adheres to the required structure.
You can find the explicit schema in `sources/model-definition-schema.json`.

Check out the declarations of known sources at `clip_eval.model.types` and refer to the existing model definitions in the `sources/models` folder for guidance.
Below is an example of a model definition for the [clip](https://huggingface.co/openai/clip-vit-large-patch14-336) model sourced from Hugging Face:

```json
{
  "model_type": "ClosedCLIPModel",
  "title": "clip",
  "title_in_source": "openai/clip-vit-large-patch14-336"
}
```

In each model definition, the `model_type` and `title` fields are required.
The `model_type` indicates the name of the class that represents the source, while `title` serves as a reference for the model on this platform.

For non-local models, the `title_in_source` field should store the title of the model as it appears in the source.
For model checkpoints in local storage, the `title_in_source` field should store the title of the model used to train it.
Additionally, on models sourced from OpenCLIP the optional `pretrained` field may be needed. See the list of OpenCLIP models [here](https://github.com/mlfoundations/open_clip/blob/main/docs/PRETRAINED.md).

### Add a Model Source

Expanding the model sources involves two key steps:

1. Create a model class that inherits from `clip_eval.model.Model` and specifies the input requirements for loading models from the new source.
   This class should encapsulate the necessary logic for processing model elements and generating embeddings.
2. Generate a model definition in JSON format and save it in the `sources/models` folder, following the guidelines outlined in the previous section.
   Ensure that the definition includes essential fields such as `model_type`, `title`, and `module_path`, which points to the file containing the model class implementation.

> It's recommended to store the file containing the model class implementation in the `clip_eval/model/types` folder and add a reference to the class in the `__init__.py` file in the same folder.
> This ensures that the new model type is accessible by default for all model definitions, eliminating the need to explicitly state the `module_path` field for models from such source.

### Programmatically Add a Model

Alternatively, you can programmatically add a model, which will be available only for the current session, using the `register_model()` method of the `clip_eval.model.ModelProvider` class.

Here is an example of how to register a model from Hugging Face using Python code:

```python
from clip_eval.model import ModelProvider
from clip_eval.model.types import ClosedCLIPModel

ModelProvider.register_model(ClosedCLIPModel, "clip", title_in_source="openai/clip-vit-large-patch14-336")
model = ModelProvider.get_model("clip")
print(model.title, model.title_in_source)  # Returns: clip openai/clip-vit-large-patch14-336
```

### Remove a Model

To permanently remove a model, simply delete the corresponding JSON file stores in the `sources/models` folder.
This action removes the model from the list of available models in the CLI, disabling the option to create any further embedding with it.
However, all embeddings previously built with that model will remain intact and available for other tasks such as evaluation and animation.

## Set Up the Development Environment

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

### Adding Dataset Sources

To contribute by adding dataset sources, follow these steps:

1. Store the file containing the new dataset class implementation in the `clip_eval/dataset/types` folder.
   Don't forget to add a reference to the class in the `__init__.py` file in the same folder.
   This ensures that the new dataset type is accessible by default for all dataset definitions, eliminating the need to explicitly state the `module_path` field for datasets from such source.
2. Open a pull request with the necessary changes. Make sure to include tests validating that data retrieval, processing and usage are working as expected.
3. Document the addition of the dataset source, providing details on its structure, usage, and any specific considerations or instructions for integration.
   This ensures that users have clear guidance on how to leverage the new dataset source effectively.

### Adding Model Sources

To contribute by adding model sources, follow these steps:

1. Store the file containing the new model class implementation in the `clip_eval/model/types` folder.
   Don't forget to add a reference to the class in the `__init__.py` file in the same folder.
   This ensures that the new model type is accessible by default for all model definitions, eliminating the need to explicitly state the `module_path` field for models from such source.
2. Open a pull request with the necessary changes. Make sure to include tests validating that model loading, processing and embedding generation are working as expected.
3. Document the addition of the model source, providing details on its structure, usage, and any specific considerations or instructions for integration.
   This ensures that users have clear guidance on how to leverage the new model source effectively.

[Falah/Alzheimer_MRI]: https://huggingface.co/datasets/Falah/Alzheimer_MRI
[trpakov/chest-xray-classification]: https://huggingface.co/datasets/trpakov/chest-xray-classification
[Kabil007/LungCancer4Types]: https://huggingface.co/datasets/Kabil007/LungCancer4Types
[sampath017/plants]: https://huggingface.co/datasets/sampath017/plants
[marmal88/skin_cancer]: https://huggingface.co/datasets/marmal88/skin_cancer
[HES-XPLAIN/SportsImageClassification]: https://huggingface.co/datasets/HES-XPLAIN/SportsImageClassification
[apple/DFN5B-CLIP-ViT-H-14]: https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14
[imageomics/bioclip]: https://huggingface.co/imageomics/bioclip
[openai/clip-vit-large-patch14-336]: https://huggingface.co/openai/clip-vit-large-patch14-336
[BAAI/EVA-CLIP-8B-448]: https://huggingface.co/BAAI/EVA-CLIP-8B-448
[patrickjohncyh/fashion-clip]: https://huggingface.co/patrickjohncyh/fashion-clip
[vinid/plip]: https://huggingface.co/vinid/plip
[flaviagiammarino/pubmed-clip-vit-base-patch32]: https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32
[flax-community/clip-rsicd]: https://huggingface.co/flax-community/clip-rsicd
[google/siglip-large-patch16-256]: https://huggingface.co/google/siglip-large-patch16-256
[google/siglip-base-patch16-224]: https://huggingface.co/google/siglip-base-patch16-224
[geolocal/StreetCLIP]: https://huggingface.co/geolocal/StreetCLIP
[wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M]: https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M
[laion/CLIP-ViT-B-32-laion2B-s34B-b79K]: https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K
[open-model-impl]: https://github.com/encord-team/text-to-image-eval/blob/main/clip_eval/model/types/open_clip_model.py
[hf-model-impl]: https://github.com/encord-team/text-to-image-eval/blob/main/clip_eval/model/types/hugging_face_clip.py
[local-model-impl]: https://github.com/encord-team/text-to-image-eval/blob/main/clip_eval/model/types/local_clip_model.py
[hf-dataset-impl]: https://github.com/encord-team/text-to-image-eval/blob/main/clip_eval/dataset/types/hugging_face.py
[encord-dataset-impl]: https://github.com/encord-team/text-to-image-eval/blob/main/clip_eval/dataset/types/encord_ds.py
