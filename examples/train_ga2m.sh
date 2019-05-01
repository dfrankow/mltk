#!/bin/bash -x

# See also the help output of the commands
# and https://github.com/yinlou/mltk/wiki/Generalized-Additive-Models

### discretize

ATTR_FILE=blob_attr.txt
TRAINING_DATA=blob_train.csv

DISC_ATTR_FILE=blob_attr.disc.txt
TRAINING_DATA_DISC=blob_train.disc.txt

java mltk.core.processor.Discretizer -r $ATTR_FILE -t $TRAINING_DATA \
     -m $DISC_ATTR_FILE -i $TRAINING_DATA -o $TRAINING_DATA_DISC

# output
TRAINING_DATA_DISC_ATTR=blob_attr.disc.txt

### train GAM

MAX_ITERS=5
GAM_MODEL=blob_train.disc.gam.model.txt
LEARNING_TASK=c

java mltk.predictor.gam.GAMLearner -t $TRAINING_DATA_DISC \
     -m $MAX_ITERS \
     -o $GAM_MODEL \
     -g $LEARNING_TASK \
     -r $TRAINING_DATA_DISC_ATTR

### train GA2M

# How do we generate a sensible PAIRWISE_TERMS_FILE instead of including all?

GA2M_MODEL=blob_train.disc.ga2m.model.txt
PAIRWISE_TERMS_FILE=blob.pairwise.txt

java mltk.predictor.gam.GA2MLearner -t $TRAINING_DATA_DISC \
     -m $MAX_ITERS \
     -g $LEARNING_TASK \
     -r $TRAINING_DATA_DISC_ATTR \
     -i $GAM_MODEL \
     -I $PAIRWISE_TERMS_FILE \
     -o $GA2M_MODEL

### Evaluate on the test set

TEST_DATA=blob_test.csv

java mltk.predictor.evaluation.Evaluator \
     -d $TEST_DATA -m $GA2M_MODEL -e AUC

# Why does Evaluator have no output?

### TODO: predict on test set
