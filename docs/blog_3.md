# Project Proposal

## Motivation

One of the biggest bottlenecks in machine learning pipelines is how data hungry they are. There have been various methods to reduce the amount of data needed to train machine learning models such as: transfer learning, active learning, and weak supervision. This project focuses on combining the above 3. The goal being, as a user hand labels a set of instances, we generate a set of functions that learn heuristics about the instances that have been labeled and apply it to the entire corpus, to get additional signal from unlabeled examples. We build on top of Snorkel [1], a machine learning platform that introduces data programming where users write python functions to label their data, and train models to learn from it. We call our approach DPD (data programming by demonstration) since this can be seen as an application of programming by demonstration in the language domain.

## Minimum Viable Plan

1. Build a supervised pipeline for NER and other sequence classification tasks with BIO encoding
2. Evaluate a benchmark with random sampling to see how this compares against dataset sizes in [1..100]
3. Investigate semi/weak supervision (presence of a small annotation set and a noisy set)
    1. Every positively annotated word gets stored in a dictionary and is in the noisy set is assumed to be true
    2. Expand the dictionary above with word embeddings to hopefully gather more data
    3. Expand the dictionary above with contextual word embeddings to hopefully be more relevant in context
    4. Investigate some structural method (described in stretch goals)

### Stretch Goals

1. Assuming the minimum viable plan goes as expected, the stretch goals are the following
2. Look into POS tags and see if patterns can be drawn there
3. Look into constituency and dependency parses
4. Look into a mix of structural and word embedding based methods

## Methodologies

## Resources

The methods we propose do not rely on large amounts of computational resources, single GPU machines or maybe even CPU machines should be sufficient in the proposed project.

## Evaluation

We will evaluate our project through comparing F1 score with amount of annotated training data for a comparison. We will compare our methods against various benchmarks to see how it performs.

## Related work

Similar work has been done in Snuba/Reef [2] for text classification through bag of word features, however bag of words is not expressive enough for sequence modeling since context of the words and order of them matters. SwellShark [3] automatically generates a series of labeling functions for Biomedical NER, which is a promising start, however it relies on access to an external knowledge base, a rather limited set of labeling functions, and has been hand tuned for the tasks at hand. We hope to extend this in a more general setting by looking at (1) a more complex DSL by taking into account linguistic features such as (POS, Constituency Parse Trees, Dependency Parses) and(2) no reliance on an external knowledge base.

## References

1. Snorkel Project: https://hazyresearch.github.io/snorkel/
2. Snorkel Reef/Snuba: http://www.vldb.org/pvldb/vol12/p223-varma.pdf
3. SwellShark: https://arxiv.org/abs/1704.06360
4. Snorkel Labeling Functions Workshop: https://www.youtube.com/watch?v=mrIkus844B4
5. Natural Language Explanation to Labeling Functions (Babble Labble Snorkel): https://arxiv.org/pdf/1805.03818.pdf
6. Random Encoders for Sentence Embeddings: https://arxiv.org/abs/1901.10444
7. CADEC Dataset: https://www.ncbi.nlm.nih.gov/pubmed/25817970
8. CONLL Dataset: https://cogcomp.org/page/resource_view/81