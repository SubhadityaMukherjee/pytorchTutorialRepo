# BGAN

- Hjelm, R. D., Jacob, A. P., Che, T., Trischler, A., Cho, K., & Bengio, Y. (2017). Boundary-seeking generative adversarial networks. arXiv preprint arXiv:1702.08431.  [Paper](https://arxiv.org/pdf/1702.08431.pdf?utm_source=aotu_io&utm_medium=liteo2_web)
- This GAN did not work without a very very low learning rate of 0.0002 so keep that in mind if you keep getting errors while running the code
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
