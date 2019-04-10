## Introduction: First Blog Post

#### team name
DPD (Data Programming By Demonstration)

#### list of members:
Akshat Shrivastava (Advised by Jeffrey Heer and Tongshuang (Sherry) Wu)

 - top three project ideas you’re excited about (Data Programming By Demonstration)
 - minimal viable action plan for each with stretch goals
 
 
#### Plans

For this project, We are continuing a research project in interactive machine learning (IML). Previously, we have worked on a system for IML for sequence modeling tasks (NER, Identifying drug reactions, etc.). A core part of this is that users have to hand label training examples (~100) to build a model to identify their sequence class. We hope to extend this, by applying Weak Supervision through Data Programming with Snorkel [1].

Snorkel introduces the concept of creating a series of label functions to assign noisy labels to create a dataset and learn from them instead. We want to investigate how we can use programming by demonstration algorithms to generate a set of labeling functions to noisily label the entire dataset and learn from that instead of a small set of gold labels provided by an end user. The idea being, that as a user labels a set of examples, we use those examples to generate a set of labeling functions and apply them to the entire dataset. The goal of the next 10 weeks would be to investigate how labeling functions can be generated and iteratively updated.

With this pipeline in mind, the main contribution would be the program synthesis component. For sequence labeling, there are two key components: (1) Identifying potential spans to label and (2) providing the identified spans with a label

##### Plan 1: Using an embedding space

Explore creating labeling functions by using an embedding space.

- Create pipeline for generating labeling functions based on a small amount of labeled examples
- MVP: iteratively build out a (or a set of) dictionary classifiers, and use a word embedding space (word2vec, glove, ELMo, BERT) to augment the dictionary
- Stretch goal: extend this dictionary to account for phrases by using sentence embedding techniques (averaging word embeddings, random encoders [6])

##### Plan 2: Using parse trees

- Create pipeline for generating labeling functions based on a small amount of labeled examples
- MVP: in conjunction to the baseline embedding space approach, identify potential spans using part of speech tags.
- Stretch Goal: instead of just POS tags, investigate how different parse trees can be used (Constitiuency Parse, Dependency Parse)

##### Plan 3: Using regex/pattern matching

- Create pipeline for generating labeling functions based on a small amount of labeled examples
- MVP: Analyze positive and negative groups to find trends in words through a series of regex matches. For example, if the goal was to identify phrases about service a rule could be: `*waiter*` to identify spans that contain the word *"waiter"*.
- Stretch Goal: implement a mix of POS tags and words. For example, we could use `ADJ waiter*` to identify spans with an adjective then waiter/waitress.

Most likely, we will implement the pipeline and a mix of the techniques described above.


#### Github Project

[https://github.com/AkshatSh/DPD](https://github.com/AkshatSh/DPD)

#### References
1. Snorkel Project: https://hazyresearch.github.io/snorkel/
2. Snorkel Reef/Snuba: http://www.vldb.org/pvldb/vol12/p223-varma.pdf
3. SwellShark: https://arxiv.org/abs/1704.06360
4. Snorkel Labeling Functions Workshop: https://www.youtube.com/watch?v=mrIkus844B4
5. Natural Language Explanation to Labeling Functions (Babble Labble Snorkel): https://arxiv.org/pdf/1805.03818.pdf
6. Random Encoders for Sentence Embeddings: https://arxiv.org/abs/1901.10444
