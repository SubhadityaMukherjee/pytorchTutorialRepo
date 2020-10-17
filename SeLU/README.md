# SELU
- Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S. (2017). Self-normalizing neural networks. In Advances in neural information processing systems (pp. 971-980). [Paper](http://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf)

- Changing ReLU -> SELU on VGG16 network
## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default config for CIFAR10 dataset.

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
