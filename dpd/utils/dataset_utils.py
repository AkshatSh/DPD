import os
import sys

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