import numpy as np 

import torch

NULL_ELEMENT = None 

def make_token_dict(vocabulary: str) -> dict:

    token_dict = {}

    vocabulary_list = list(vocabulary)
    vocabulary_list.sort()

    token_dict[NULL_ELEMENT] = np.array(0).reshape(-1,1)
    for token, element in enumerate(vocabulary):
        token_dict[element] = torch.tensor(np.array(token + 1)).long().reshape(1,-1)

    return token_dict

def tokens_to_one_hot(tokens, pad_to=100, pad_classes_to=33):

    one_hot = torch.zeros(*tokens.shape[:-1], pad_classes_to)

    for ii in range(tokens.shape[0]):
        one_hot[ii, tokens[ii,:].long().item()] = 1.0

    for jj in range(ii, pad_to):

        one_hot[jj, 0] = 1.0

    return one_hot 

def sequence_to_vectors(sequence, sequence_dict, pad_to=64):

    vectors = None

    for element in sequence:

        if vectors is None:
            vectors = sequence_dict[element]
        else:
            vectors = np.append(vectors, sequence_dict[element], axis=0)

    while vectors.shape[0] < pad_to:
        vectors = np.append(vectors, sequence_dict[NULL_ELEMENT], axis=0)

    return torch.tensor(vectors, dtype=torch.float32)

def one_hot_to_sequence(one_hot, sequence_dict):

    sequence = ""

    key_dict = {sequence_dict[key].item(): key for key in sequence_dict.keys()}
    for element in one_hot:

        index = torch.argmax(element).long().item()
        if index in key_dict.keys():
            sequence += key_dict[index] if index != 0 else ""
        else:
            print(index, "key not in dict")
            sequence += "" #key_dict[list(key_dict.keys())[0]]

    return sequence
