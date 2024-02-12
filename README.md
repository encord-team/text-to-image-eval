## Goals are:

- [ ] Have a way to load classification datasets from HF. See this [colab](https://colab.research.google.com/drive/1O7PBYHKrk8SELHq40AoH8hehig-WNezS?usp=sharing)
- [ ] Have a way to load clip mod from HF. See this [colab](https://colab.research.google.com/drive/1O7PBYHKrk8SELHq40AoH8hehig-WNezS?usp=sharing)
- [ ] Find relevant datasets for medial domain
- [ ] Find relevant "CLIP" models for medical domain
- [ ] Compute embeddings across datasets and models for medical and store them
- [ ] Evaluate each model on each dataset based on the `evaluation` module in this repo

- Repeat for geospatial and sports analytics

### Commands I used to run different bits of the code

0. data models: `PYTHONPATH=$PWD/src python src/types/data_models.py`
1. knn: `PYTHONPATH=$PWD/src python src/evaluation/knn.py`
2. zero shot: `PYTHONPATH=$PWD/src python src/evaluation/zero_shot.py`
3. linear probe: `PYTHONPATH=$PWD/src python src/evaluation/linear_probe.py`
4. evaluation: `PYTHONPATH=$PWD/src python src/evaluation/evaluator.py`
