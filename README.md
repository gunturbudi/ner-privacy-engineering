# A Named Entity Recognition Based Approach for Privacy Requirements Engineering

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5801370.svg)](https://doi.org/10.5281/zenodo.5801370)


The code, dataset and best model from the Paper: "A Named Entity Recognition Based Approach for Privacy Requirements Engineering"

Download the best model trained on BERT with WordNet Synonym Augmentation: https://s.id/Ta7k

Then you can predict the privacy entities:

```{python}
from flair.data import Sentence
from flair.models import SequenceTagger

# load the trained model
model = SequenceTagger.load('ner-model.pt')

# your user story
sentence = Sentence('As an OlderPerson, I want to know exactly what ALFRED does with my personal data, and share it only on my specific permission.')

# predict the tags
model.predict(sentence)

print(sentence.to_tagged_string())

```

Please kindly consider citing the following papers if you find this repository useful for your research.

```
@inproceedings{9582331,
  author = {Herwanto, Guntur Budi and Quirchmayr, Gerald and Tjoa, A Min},
  booktitle = {2021 IEEE 29th International Requirements Engineering Conference Workshops (REW)},
  doi = {10.1109/REW53955.2021.00072},
  pages = {406--411},
  title = {{A Named Entity Recognition Based Approach for Privacy Requirements Engineering}},
  year = {2021}
}
```
