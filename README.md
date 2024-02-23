## Goals are:

- [ ] Have a way to load classification datasets from HF. See this [colab](https://colab.research.google.com/drive/1O7PBYHKrk8SELHq40AoH8hehig-WNezS?usp=sharing)
- [ ] Have a way to load clip mod from HF. See this [colab](https://colab.research.google.com/drive/1O7PBYHKrk8SELHq40AoH8hehig-WNezS?usp=sharing)
- [ ] Find relevant datasets for medical domain
- [ ] Find relevant "CLIP" models for medical domain
- [ ] Compute embeddings across datasets and models for medical and store them
- [ ] Evaluate each model on each dataset based on the `evaluation` module in this repo

- Repeat for geospatial and sports analytics

### Set up the development environment

1. Create the virtual environment, add dev dependencies and set up pre-commit hooks.
   ```
   ./dev-setup.sh
   ```
2. Add environment variables:
   ```
   export CLIP_CACHE_PATH=$PWD/.cache
   export OUTPUT_PATH=$PWD/output
   export ENCORD_SSH_KEY_PATH=<path_to_the_encord_ssh_key_file>
   export ENCORD_CACHE_DIR=$PWD/.cache/encord
   ```

### CLI Interface

Basic CLI interface available with:
`PYTHONPATH=$PWD python src/cli.py`


### Commands I used to run different bits of the code

0. data models: `PYTHONPATH=$PWD python src/common/data_models.py`
1. knn: `PYTHONPATH=$PWD python src/evaluation/knn.py`
2. zero shot: `PYTHONPATH=$PWD python src/evaluation/zero_shot.py`
3. linear probe: `PYTHONPATH=$PWD python src/evaluation/linear_probe.py`
4. evaluation: `PYTHONPATH=$PWD python src/evaluation/evaluator.py`
