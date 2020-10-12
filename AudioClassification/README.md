# Audio classification using a simple network

- Dataset used [Urban8k](https://urbansounddataset.weebly.com/urbansound8k.html)
- Reference article [Link](https://medium.com/@aakash__/classifying-audio-using-pytorch-84861f3505ea)

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default 

- If you use "my" for the architecture, it will take it from Nets.py

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
