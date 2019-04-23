# Algorithms

In this markdown page I write the pseudo code for various algorithms I implement in this pipeline

## Active Learning

The algorithm below describes the generic active learning set up I am working with, there are comments
above each line as well to explain each one.

```python
heuristic = RandomHeuristic()

# the current training data that is being built up
train_data: DatasetType = []

# build the validation set and load into memoery
valid_dataset = BIODatasetReader(...)

oracle_samples = [1, 5, 10, 25, 50, 100, 200, 400, 400]
for i, sample_size in enumerate(oracle_samples):
    # select new points from distribution
    distribution = heuristic.evaluate(unlabeled_dataset)
    query = torch.multinomial(distribution, sample_size)

    oracle_labels = [oracle.get_query(q) for q in query]

    # extend the oracle labels
    train_data.extend(oracle_labels)

    # remove labeled data points from unlabeled corpus
    unlabeled_dataset.remove(q)

    model = train(
        train_data=train_data,
        valid_data=valid_dataset,
    )
```

## Weak Active Learning

This is an extension to the algorithm above, namely it adds this aspect of building a `weak dataset` and using the 
weak dataset along with the train dataset for training.

```python

def build_weak_data(
    train_data,
    unlabeled_corpus,
    model,
):
    # use the model and annotated data to build some
    # heuristics
    heuristics = generate_heuristics(train_data, model)

    # apply each heuristic and store the new labeled data set
    # if h heuristics, stores h versions of the annotated dataset
    h_annotations = []
    for h in heuristics:
        h_ann = h.apply(unlabeled_corpus)
        h_annotations.append(h_ann)
    
    # merge all the annotations
    weak_labels = collate(h_annotations)

    return weak_labels

```

The adjustments to the active learning algorithm to account for this weak set are shown below

```python
heuristic = RandomHeuristic()

# the current training data that is being built up
train_data: DatasetType = []

# build the validation set and load into memoery
valid_dataset = BIODatasetReader(...)

oracle_samples = [1, 5, 10, 25, 50, 100, 200, 400, 400]
for i, sample_size in enumerate(oracle_samples):
    # select new points from distribution
    distribution = heuristic.evaluate(unlabeled_dataset)
    query = torch.multinomial(distribution, sample_size)

    oracle_labels = [oracle.get_query(q) for q in query]

    # extend the oracle labels
    train_data.extend(oracle_labels)

    # Create a weak set based on the training data
    # and the model and the unlabeled corpus
    weak_data = build_weak(
        train_data=train_data,
        unlabeled_corpus=unlabeled_corpus,
        model=model,
    )

    # remove labeled data points from unlabeled corpus
    unlabeled_dataset.remove(q)

    model = train(
        train_data=train_data + weak_data,
        valid_data=valid_dataset,
    )
```