# WGAN GP

- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. In Advances in neural information processing systems (pp. 5767-5777). [Paper](https://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans.pdf)
- (Original GAN)
- Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein gan. arXiv preprint arXiv:1701.07875.
[Paper](https://arxiv.org/pdf/1701.07875.pdf%20http://arxiv.org/abs/1701.07875
)
- Code for a WGAN and generating images using it
- Note, GANs are hard to train
    - It might take too long if your data is very varied (aka just one or two types of things vs 10 or 100)
    - Try to choose better hyper params. (Especially high learning rate will kill the gradients very fast)

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.002 --log-interval 100 --arch "my"
```
will run default  on MNIST (I would do CIFAR but I want it to run faster and im sleepy)

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
