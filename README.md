# A Named Entity Recognition Based Approach for Privacy Requirements Engineering
The code, dataset and best model from the Paper: "A Named Entity Recognition Based Approach for Privacy Requirements Engineering"

Download the best model trained on BERT with WordNet Synonym Augmentation: https://s.id/G6iaR

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
