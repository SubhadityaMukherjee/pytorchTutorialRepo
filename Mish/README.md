# Mish

- Misra, D. (2019). Mish: A self regularized non-monotonic neural activation function. arXiv preprint arXiv:1908.08681. [paper](https://arxiv.org/pdf/1908.08681)
- Changing ReLU -> Mish on VGG16 network
## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default config for CIFAR10 dataset.

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
