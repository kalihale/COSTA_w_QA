# Runs the training
# -lr passes a number of training rates and -e passes the number of epochs
python3 training.py -lr 5e-4 5e-5 5e-6 5e-7 -e 4

# Runs the testing
python3 testing.py -lr 5e-4 5e-5 5e-6 5e-7 -e 4

# Starts tensorboard to view loss rates, go to http://localhost:6006 to view tensorboard
tensorboard --logdir=./runs