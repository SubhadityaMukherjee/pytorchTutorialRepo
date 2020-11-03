# Coupled GAN

- Liu, M., & Tuzel, O. Coupled Generative Adversarial Networks. arXiv 2016. arXiv preprint arXiv:1606.07536. 
- [Paper]( https://arxiv.org/abs/1606.07536) 

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
