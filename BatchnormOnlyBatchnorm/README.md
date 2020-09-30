# Training batchnorm and only batchnorm

- Pytorch implementation of this [paper](https://arxiv.org/abs/2003.00152) 

- @misc{frankle2020training,
      title={Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs}, 
      author={Jonathan Frankle and David J. Schwab and Ari S. Morcos},
      year={2020},
      eprint={2003.00152},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

## What is it?

- Just train batchnorm layers and freeze others

## How to run?

### Default

```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "resnet18"
```
will run default 

- If you use "my" for the architecture, it will take it from Nets.py

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
