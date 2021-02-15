- https://www.kaggle.com/sherinclaudia/movielens

# Notes

- Not much changes between multiclass, multilabel and binary

## General

- Pandas use engine python to prevent some weird errors
- Bert is a pretty huge model :/ Albert is smaller and better

## Binary
- BCE With logits loss and target.view(-1,1) works.
- Number of classes is 1

## Multiclass
- No of classes is 2
- Cross entropy

## Multilabel
- The returning data dictionary changes to allow for all classes (so we can return probabilities oops)
