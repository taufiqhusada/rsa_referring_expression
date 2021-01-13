Find out images with very low percentage match
Compare that against the images with 0 bleu score.

##Note Oct 22:##
Progress:
# Run scenegraph on pc7
# Try synonyms with nltk.wordnet

TODO:
## DATA PROCESSING & OUTPUT MEASURE
# synonyms with nltk wordnet is incomplete.
-> need related words, not just synonyms. Try Spacy?
-> try meteor score metric => give synonyms score.
-> BERT score

## RSA IMPROVEMENT
# incorporate relations to the rsa model
-> extracting the most likely relation from rel_<id> tables  #### DONE ####

# currently the expressions contain 1 object follow by some words of type attributes
-> Shorten the # of attributes that follow a type/object. 
-> add relation(s):
=> reverse(type + (some) attribute ) + relation
EG: (big green apple ) (next to) (cup)

# adding more relations beside rel_tables:
-> adding spacial/order relation between the bounding boxes.
=> Apple example: compare the apple boxes: 3rd box from left.

### salience in the attributes table is created based on size.
? Use salience to create relation that compare size of objects ? 'smaller', 'larger', 'largest'

##Note Nov 11:##
Progress:
# Get attributes & rel for training set (finally)

# Incorporate Meteor metric 

# WIP: add LSTM with labels as the training data

framework:
1. Generating Utterances & objects:
    + scenegraph & RCNN
    + add relation ?
2. Generate expression:
    + RSA (+ relation ?)
    + LSTM ?

ideas: add ordering/size relations
- count # of appearances of a type
- if # > 1 => add relation: 2nd from left, bigger/biggest obj

##check the training data. calculate the statistics of these relations (3 types: between, among, ordinal).
parsing sentences to get a form (N - Relational term - N) -> get a relation.

how often do they appear (next to, has, ...). Find landmark, how many? how salient is the landmark (big or small). Add a ordering/preferences to relations - prefer the bigger?

# 11/18

## 27264 out of 40,000 images have target labels containing spatial relation terms ['left', 'right', 'top','bottom', 'under', 'above']


## try higher thresholding the likelihood of each type/attributes 

## visualize the updated prior after each step of RSA.

## 12/1 
Ideas: list of attributes & relations (spatial). Learn when to use attributes or relations ? Graph to sequence.
https://arxiv.org/pdf/1804.00823.pdf
Applying graph2Seq technique to generate referring expression.
code
https://github.com/IBM/Graph2Seq

another paper with transformer https://www.aclweb.org/anthology/2020.acl-main.640.pdf 


