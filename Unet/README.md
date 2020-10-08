# Dynamic Unet for semantic segmentation (This is a work in progress)

- [Paper](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Paper2](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- Inspired a lot by [usuyama](https://github.com/usuyama/pytorch-unet)

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.01 --log-interval 100 --arch "my"
```
will run training

To run inference run
```bash
python3 example.py --input_image "./img.jpeg" --model "./models/model.pt" --output_filename "./outputs/output.jpg" --cuda 
```

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
