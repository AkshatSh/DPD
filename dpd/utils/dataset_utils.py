from typing import (
    List,
    Tuple,
    Dict,
    Iterator,
)
import string

from dpd.dataset import BIODataset, BIODatasetReader

from dpd.constants import (
    STOP_WORDS,
    CONLL2003_TRAIN,
    CONLL2003_VALID,
    CONLL2003_TEST,
    CADEC_TRAIN,
    CADEC_VALID,
    CADEC_TEST,
)

def get_dataset_files(dataset: str) -> Tuple[str, str]:
    '''
    returns train_file, valid_file in BIO encoding
    input:
        ``dataset`` str
            the dataset type specified in a string in {CONLL, CADEC}
    output:
        ``Tuple[str, str]``
            the train file and valid file
    '''
    if dataset == 'CONLL':
        return CONLL2003_TRAIN, CONLL2003_VALID, CONLL2003_TEST
    elif dataset == 'CADEC':
        return CADEC_TRAIN, CADEC_VALID, CADEC_TEST
    else:
        raise Exception(f'Unknown dataset: {dataset}')

def remove_bio(tag: str) -> str:
    '''
    input:
        tag: str, input tag
    output:
        tag: str, remove the BIO encoding e.g. B-Arg becomes Arg and
        I-Arg becomes Arg
    '''
    if tag[0] in {'B', 'I'}:
        return tag[2:]
    return tag

def get_words(sentence: List[str], tags: List[str], bin_tag: str) -> List[str]:
    output = []
    for word, tag in zip(sentence, tags):
        r_tag = remove_bio(tag)
        if word in STOP_WORDS or r_tag != bin_tag or word in string.punctuation:
            continue
        output.append(word)
    return output

def explain_labels(
    example: List[str],
    seq_label: List[str],
) -> Tuple[List[Tuple[int, int]], List[str]]:
    '''
    Convert a label to a list of word ranges and entities

    word_range[i] = (start: int, end: int) with end exclusive
    entities[i] = str, the entity corresponding to word_range[i]
    '''
    ranges : list = []
    entities: list = []
    range_start : int = None
    seq_label = [] if seq_label is None else seq_label
    for i, label in enumerate(seq_label):
        if (label == 'O' or i == len(seq_label) - 1) and range_start is not None:
                ranges.append(
                    (
                        range_start,
                        i,
                    )
                )
                entities.append((example[range_start : i]))
                range_start = None
        elif label.startswith('B'):
            if range_start is not None:
                ranges.append(
                    (
                        range_start,
                        i,
                    )
                )
                entities.append((example[range_start : i]))
            range_start = i

    return ranges, entities