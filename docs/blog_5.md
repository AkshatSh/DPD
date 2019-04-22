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

5. **Evaluation Metric F1**

This was left out from the previous blog post, but the evaluation metric is a **Token** level F1 score instead of a span F1 score. The rationale here is that in the CADEC dataset, the spans can get a few tokens long, and if a single token is left out of the span, this is considered to be a false positive and can hurt the F1 score quite a bit. Instead we take a look at the token level F1 score. If you want to see the Sppan F1 score, I am happy to report that as well.

## Other random updates

Since this is a continuation of another research project, part of the work this quarter is to port the entire modeling side to AllenNLP instead of the custom modules I have been writing. Just this week I was able to replicate some of my experiments (specifically the supervised ones) using AllenNLP instead of my custom modules. I will continue to work on this so everything will be built on top of AllenNLP, however if you look at the actual GitHub and notice some things are missing/incomplete this would be why. Most experiments have been done on my private research Repo, but should be migrated here soon.

## Baseline Descriptions

### Keyword Matching

As mentioned in the [previous blog post](blog_4.md) one of the baseline approaches, I took a look at was simple keyword matching. In particular, if we have our training data T, extract all the positively labeled words in T, and label them as positive in our noisy set and do some weighted training.

### Phrase Matching

This is a continuation of the above approach, however instead of looking at words we will look at phrases. In particular now  if `leg pain` was an entity `leg` and `pain` would be classified as positive in the keyword matching baseline, however in phrase matching only `leg pain` in that order will count as a match.

### GLOVE embedding space

The limitations of the keyword and key phrase matching approaches is that they do no generalize to unseen words, which is a rather large limitation given our set of positively annotated words can be quite small.

We look to overcome this limitation by looking at expanding this set of positively annotated words by using an embedding space to augment this dictionary of positve words. In particular we take a look at using a `kNN` approach, then `logistic regression`, and finally `SVM`. Descriptions and rationale for each are listed in the associated sections below.

#### kNN

Using `FAISS` [1 Johnson et al. 2017], we index all the GLOVE word embeddings (with embedding dim `d`). Then we use our dictionary of positively labeled words to form a query which contains the embedding vectors for each of the words in the dictionary (shape `(num_words, d)`). We then search for the closest `k` vectors using `cosine similarity` as our similarity metric.

The result we get back is `(num_words, k)` giving the `k` closest vectors for each word in the query. We then convert this to a ranked list where we represent our similar words as `(word, count)` where count is the number of times the word appears in our result matrix.

#### Logistic Regression

When analyzing the results of the `kNN` approach, we find that while most words are relevant, some of them are not. We suspect this is because the concept we want to capute (e.g. `Adverse Drug Reactions`) for our dataset may be similar in some dimensions and different in others, which could potentially cause the `kNN` approach to fail.

To overcome this instead we take all our positive words, and sample our negative words to create a training set. Where `w_i` has an embedding vector `e_i` and an associated label `l_i`, where `l_i = 1` if `w_i` is positive and `l_i = 0` if `w_i` is negative. We then create a training set where the input is `e_i` and the output is `l_i`, and train a logistic regression model.

Once our logisitc regression model is trained, we run the model over all the words in the glove embedding space, and conver this to a ranked list of similar words as `(word, prob)` where prob is the probability that the word belongs to the positive class.

*This is all implemented using `Sklearn`*

#### SVM

Similar to the approach above we attempt to the same algorithm except replacing the logistic regression model with an SVM of different kernels.

- Linear Kernel
- RBF Kernel
- Quadratic Kernel

## Performance Report

### Sampled Dictionaries
| kNN                                                                                                                                                           | Logistic Regression                                                                                                                                                                                                                                                                                                                                  | SVM: Linear                                                                                                                                                                                                                                                                                                                                     | SVM: RBF                                                                                                                                                                                                                                                                                                                                        | SVM: Quadratic                                                                                                                                                                                                                                                                                                                           |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [ ('ankle', 7), ('vomiting', 6), ('shortness', 6), ('groin', 4), ('anxiety', 4), ('hamstring', 3), ('elbow', 3), ('maybe', 3), ('worried', 3), ('lips', 3), ] | [ ('numbness', 0.9174100396266176), ('rashes', 0.9012706273738842), ('dizziness', 0.8934524597467283), ('protruding', 0.890337491119129), ('bruised', 0.8853579871679472), ('irritability', 0.8801254161047158), ('itchy', 0.879740547126885), ('swollen', 0.8773934862144145), ('lethargy', 0.8740313251993611), ('blisters', 0.8736541852951585) ] | [ ('numbness', 0.9355321308207033), ('dizziness', 0.9298693708907504), ('rashes', 0.9258616378965004), ('blisters', 0.9239482922376012), ('faint', 0.9170416074468241), ('itching', 0.9137676947900337), ('bruised', 0.9120761829930291), ('tingling', 0.9068303050285151), ('slurred', 0.9068260885147693), ('coughing', 0.9067132605655276) ] | [ ('numbness', 0.9808175304349438), ('dizziness', 0.9765631795707317), ('rashes', 0.9711353898062915), ('blisters', 0.9672192791508427), ('nausea', 0.9668084543325492), ('cramps', 0.9664933184886196), ('cramping', 0.9654121794104947), ('twitching', 0.9650861084373981), ('vomiting', 0.9617692471965692), ('aches', 0.9608794842985615) ] | [ ('dizziness', 0.9444865908041784), ('nausea', 0.938131957998582), ('numbness', 0.9276490899050369), ('headaches', 0.9153989333560578), ('vomiting', 0.9126958388894453), ('cramps', 0.9102443871543957), ('aches', 0.900947414528268), ('cramping', 0.8815301370725462), ('sore', 0.8764006132712062), ('aching', 0.875818424526165) ] |

### Augmented Dictionary Top 10 items

### Active Learning Graphs

## Error Analysis

## References

1. FAISS (Facebook AI Simlarity Search)
    - Johnson, Jeff and Douze, Matthijs and J'egou, Herv'e
    - 2017 arxiv
    - [paper](https://arxiv.org/abs/1702.08734)
    - [github](https://github.com/facebookresearch/faiss)