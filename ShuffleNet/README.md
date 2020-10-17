# ShuffleNet

- Zhang, X., Zhou, X., Lin, M., & Sun, J. (2018). Shufflenet: An extremely efficient convolutional neural network for mobile devices. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 6848-6856). [Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf)

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default config for CIFAR10 dataset.

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
