# COSTA for QA

By Kali Hale and Setareh Najafi Khoshnoo

## Run instructions

To run as-is, run `run.sh`.

A number of learning rates can be passed with the `-lr` option.

The number of epochs (an int) should be passed with `-e`.

`run.sh` defines the number of epochs and the learning rates that were
used in our project.

## Metrics output

The loss is output to the `/runs` folder using Tensorboard and SummaryWriter.
The information can be viewed live by running the command `tensorboard --logdir=./runs`
in the local directory, or `run.sh` will run the command when the training and testing
is done.

The test results are printed to the console after testing of each model is complete.

The GitHub repository is at https://github.com/kalihale/COSTA_w_QA
