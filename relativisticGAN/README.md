# Relativistic GAN ; Relativistic Average GAN (RGAN, RaGAN)

- Jolicoeur-Martineau, A. (2018). The relativistic discriminator: a key element missing from standard GAN. arXiv preprint arXiv:1807.00734. 
[Paper]( https://arxiv.org/abs/1807.00734 )
- Note, GANs are hard to train
    - It might take too long if your data is very varied (aka just one or two types of things vs 10 or 100)
    - Try to choose better hyper params. (Especially high learning rate will kill the gradients very fast)

- Almost the same as "draGAN"

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.002 --log-interval 100 --arch "my"
```
will run default  on MNIST (I would do CIFAR but I want it to run faster and im sleepy)

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
