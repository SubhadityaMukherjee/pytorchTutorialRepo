# Squeeze Net

- SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 1MB model size (2016), F. Iandola et al.
[Paper](http://arxiv.org/pdf/1602.07360)

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default config for CIFAR10 dataset.

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
