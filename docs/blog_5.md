# More Baslines and Analysis!

## Something from the last blog post

There were somethings left unclear in the last blog post, so I wanted to address them here.

1. **What is dataset size and how does this compare with the noisy set and the gold set?**

The dataset size refers only to the gold set size. The way I have it implemented right now, we have a labeled and unlabeled corpus where the two of them put together is the total number of training instances avaiable in CADEC (e.g. `1000`). So when the dataset size is at `100`, this would mean that the true dataset size is `100` and the unlabeled set is `900`. The noisy set is constructed by applying our heuristics to the unlabeled corpus, not the union, so it is not fixed.

Maybe I should use the entire dataset for the noisy set including already labeled instances, not just the unlabeled corpus. I haven't thought of that, but it may help.

The purpose of the dataset size being the number of gold instances is simply to give a proxy to annotation cost, that on average if someone spent time labeling 50 instances or 100 instances, from emprical results this could be their gain.

2. **User studies**

For the purpose of this project, I will use already labeled datasets and simulate an active learning environment, by creating an unlabeled corpus by simply removing those labels and allowing an annotator object to just provide those labels to the system when requested. This also gives a good abstraction to experiment with noisy annotators (what if only 90% of the time the annotation was correct), if we chose to go down that route.

3. **The use of structural information**

A part of the labeling functions is to provide labels to spans of text. However, a part of this problem is *what spans of text should we consider candidates?* Well, we could look at all n-grams with n up to 6 or 7, however this could get quite computationally expensive. My intention was to use the structural information like POS tags and parse trees to prune the set of potentially spans to look at to some feature based candidate extractor. I note it was pointed out that these taggers can be noisy especially on user authored text like the CADEC dataset would be, however the hope is that since this dataset is intended to be noisy and we are just trying to prune candidates, this won't be too bad. However, this is part of the reason I will also take a look at the CoNLL 2003 task for NER, since identifying mentions of people in text, should be rather trivial as compared to drug reactions, since most mentions of people should be `nouns`.

Another interesting point made to me during my presentation was instead of just pruning candidates with this structual information, maybe use it in some way for vector composition on the contextual embeddings, and this approach could help with creating phrase embeddings as well.

4. **Metadata**

One of my classmates, `Byran Hanner`, mentioned adding some form of metadata on top of text features. It would be really interesting to see how features outside of text could help in generating this noisy set. However, I want the system to be as general as possible, and am not quite sure what assumptions I could make on the availability on metadata, maybe some generic knowledge base such as Freebase or DBpedia could help in this. While it may not be something I can focus on given we only have 6-7 weeks left, it is definetly something on my list of things to explore.

## Baseline Descriptions

### Keyword Matching

### Phrase Matching

### GLOVE embedding space

## Performance Report

## Error Analysis