# Continue Advance Solution #1

[Blog Main Page](README.md): has links to all the previous blog posts.

## Addressing things from the last blog post

### POS functions

Part of my comments were `The word/span -> positive selection for labeling functions is fairly intuitive; do you think this intuition will extend to POS tags/parses?`

Reponse: I think (and the way I have been working on this) was to directly extend the intuition. While this may not be clear as to why because predicting something like an adverse reaction soley based on the POS/parse features is not going to work, it will be used along side many other labeling functions and weak classifiers. So while it may not be perfect, it provides some supervision as to what our model should be learning from this noisy set.

## Group Feedback

I am working by myself (at least in the context of this class), so I will use this area to mostly reflect on what has been going on this quarter. While this project does look quite large from the outside, I want to mention that the entire Active Learning component of this pipleline and modeling components have been done in previous work before this class. That being said, what I am finding is that the number of things to try for this project is growing larger and larger, and may habe been a bit ambitious for me to complete in 10 weeks. With that in mind, I do think I have been progressing quite well with the experimentation and engineering work. To keep this pace up, it will be important for me to decide one path to focus on for the rest of the quarter, and I will explain that direction towards the end of this post.

## Continuing Advanced Solution

In my [previous blog post](blog_6.md) I mentioned that the advanced solution I was working on was collation of weak functions. In particular, how can we combine multiple labeling functions produce a stronger weak set. Parallely, I also was working on incoporating various new weak functions into my pipline and experimenting with them. The challenge here became a lot of engineering work ontop of experimentation too. In this section, I will go over the engineering challenges I was working on, and the various experiments I ran.

### Labeling Function Generating Pipelines

After some thought the set up for generation of a labeling function is the following:

![lf_diagram](figures/lf_diagram.png)

* **Feature Extractor**: Is what converts each sentence into a (sentence_len, embedding_dim) matrix of features. The ones I have implemented are: (`ELMo`, `BERT`, `GloVe`, `word (one hot)`, `POS tags`).
* **Feature Summarizer**: Suppose a function takes in a series of features, for example the window function takes `window_width` number of features for each instance. Feature summarizer is a simple transform that decides how these features are combined. This could be either `sum` to add all the features together or `concat` to concatenate them together, to retain some information about order.
* **Weak Function**: This is actual function to apply. For example, `exact match` checks to see if the incoming feature has been seen before, and if it has assigns it a label. Linear trains a `svm_linear` classifier to classify positive and negative terms in the training data. `kNN` does the same thing except with the `kNN` algorithm. Lastly, window takes a `window_dim`, and uses the context of `window_dim` items before the current word and `window_dim` items after the current word to classify the current one.

Enumerating a subset of all posibile combinations of these we end up with around 30 labeling functions we need to train and apply.

### Engineering Work

Now that we have 30 labeling functions, at each iteration we need to train them on the training sample we have (`10`, `50`, and `100`). Each of these functions most likely trains either an SVM or kNN classifier at the word level and applies it to the entire unlabeled corpus (at minimum `900` unlabeled instances). We use `scikit learn` to implement these however, each classifier takes around `1-2 minutes` to train, and increases drastically as the training set size increases. Doing this in sequence, would mean easily `0.5-1.5 hours` extra time to compelete each iteration, which is quite slow. To overcomb this, I implemented a multiprocessing pipeline that parallelizes this work across the CPU, which definetly helps, but there are some memory contraints that come up. In particular if I ask a different process to execute an `SVM` classifier over `ELMo vectors` the `ELMo` module gets copied over to the other process and now there are 2 `ELMos` in RAM (same with `BERT`), this aggregates quickly and the system can run out of memory (don't worry, I am not using the NLP capstone GPU machines for this, so this shouldn't bother anyone besides me).

Messing with this limitations, I have brought down the runtime from `0.5-1.5 hours` to `10-30 minutes`, which is a substantial increase, but still more work can be done here.

### Results from Active Learning Experiments

The results from the latest experiment are presented here. There is still quite a bit of work to be done on the experimentation side, which I will continue doing, but to briefly talk about what the issues are and where I plan to go.

### Error Analysis

#### Labeling Function Analysis

Now part of `Snorkel` allows us to analyze the labeling functions we have trained. In particular look for what classes they are predicting, how much they overlap with other labeling functions, how much of the data they cover, and how much they conflict with other labeling functions. The results are presented below.


In particular notice how coverage and overlaps are `1.0`, this is because the way I wrote each function was to predict whether a word was positive or negative. However, digging into this further, `snorkel` heavil relies on having one function predict whether a word is positive and if it says its not, assign a `VOID` label not a negative one, and have a separate function determine if something is negative or not. A part of what I plan to do the next couple of days is to rewrite the labeling functions to follow this paradigm, and hopefully we will see some interesting results from that.

## Next Blog Post / Next Solution

Now that we have this infrastructure in place, and a good experimenting pipeline for these autogenerated weak functions/classifiers. The next thing to do is focus on how to train them.

### Continue work on this

As mentioned earlier, I will continue working on some of the problems I described in the error analysis that have clear fixes.

### Next Big Thing to Try

Weighted Training or `noisy -> gold` training don't seem to fully capture the noisy set. There are a few ideas, I have to leverage this better. The main theme is instead of treating the noisy set as a noisy version of the gold set. Treat it as a different but related task. This could change the goal from using the noisy set to further solidfy model predictions, to instead use the noisy set to refine the hidden states of the model.

#### Freeze and Retrain

#### Multitask Learning

#### Unsupervised Data Augmentation

##### Paraphrasing

`TODO: summarize paper`

`TODO: include figure`

* Backtranslation
* Parabank