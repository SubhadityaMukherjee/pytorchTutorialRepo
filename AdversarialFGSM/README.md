# Adversarial Examples -> Fast Gradient Sign Method

- Firstly, look at this [Blog](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)
- Secondly, I will post my interpretation of this + paper soon

- The objective is to run adversarial examples on MNIST and understand how they work. 
- Major changes are only in the tester.py file. Rest remains the same

## How to run?

### Default

Train the network first
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default 

- Then run 
```bash
python3 tester.py
```

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.
