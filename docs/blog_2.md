## Second Blog Post

### Pros and Cons of each Idea

From the previous blog post, we identified 3 directions to go down for generating labeling functions


#### Embedding space

**Pros**:

- Investigates how contextual embeddings can be used in low resource settings, and what representations can be generalized from them

- Does not need many computational resources

- Potentially simple to implement

**Cons**:

- Building out a dictionary may not be innovative

- May be potentially too simplistic

#### Structural information

**Pros**:

- More novel

- Analyzing and grouping structures could be an important step in generating these labeling functions

**Cons**:

- May not necessairly leverage new things in the NLP domain

- Could potentially be computational expensive to analyze a tree and group them for every sentence in a dataset

#### Regex/Pattern Matching

**Pros**:

- Mixes some structural information and word information to provide potentially stronger functions

**Cons**:

- Could be hard to generalize

- May be quite difficult to explore a rather large combinatorial space

### Final Plan

With the information from above, we think the best would be to implement a mix of the ideas above. Using an embedding space with a dictionary is a good first step in order to incorporate some domain knowledge, and then investigate how to further use the embedding space or look into structural information depending on initial results.

Current Plan:

1. Build out a generic pipeline, that allows us to iteratively evaluate model performance as a dataset grows.

2. Hard code some functions to use Snorkel for applying the same pipeline to a noisy dataset instead of gold one.

3. Look into building an augmenting a dictionary through the labeled instances in the dataset, and use this to generate the first set of labeling functions

4. Compare this method against baselines (Snuba/Reef, AutoNER)

5. Look into incorporating structural information by analyzing different parse trees and POS tags of the labeled instances, and use this to generate the next set of labeling functions

6. Compare this method against baselines

### Codebases

We will build our models in PyTorch [1] and potentially using AllenNLP [6]. We will build our system ontop of Snorkel [2]. We will compare our implementation against Snuba/Reef [3] and AutoNER [5], and use Snorkel MeTal (a implementation of Snorkel for GLUE) [4] to help with writing models for sequence classification.


### Lecture

A lecture on important linguistic features for different NLP tasks would be useful.

### References

1. PyTorch: https://pytorch.org/
2. Snorkel: https://github.com/HazyResearch/snorkel
3. Snorkel Snuba/Reef: https://github.com/HazyResearch/reef
4. Snorkel MeTal: https://github.com/HazyResearch/metal 
5. AutoNER: https://github.com/shangjingbo1226/AutoNER
6. AllenNLP: https://allennlp.org/