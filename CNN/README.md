# cut-marks classification using CNNs

## Intro
A TensorFlow implementation of the paper: W. Byeon, M. Domínguez-Rodrigo, G. Arampatzis, et al., "Automated identification and deep classification of cut marks on bones and its paleoanthropological implications", Journal of Computational Science, 2019.
 
## Requirements
- Python 3
- Tensorflow GPU

## Training/testing the model
see run-train.sh
```shell
python CNN.py [options]

```
options
```shell
python pyramid.py \ 
--train \ # if training 
--save \ # save the model
--resume \ # resume the trained model
--real_test \ # if using the real dataset
--use_fp16 \ # use half floats: True or False (default)
--do_dropout \ # use dropout: True or False (default) 
--do_weight_decay \ # use weight decay: True or False (default) 
--num_runs \ # the number of runs: default 1
--model_fname \ # model filename for saving
--vis \ # visualization type: gradcam, activations, or filters
```

## Contacts
This code was written by [Wonmin Byeon](https://github.com/wonmin-byeon) (wonmin.byeon@gmail.com).

Please open an issue for code-related questions or reporting issues. 
