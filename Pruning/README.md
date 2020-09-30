# Pruning

- [Blog](https://medium.com/datadriveninvestor/reducing-model-footprint-in-deep-learning-c500b3ff50b)

- This is a demo of model pruning
- Outputs for the pruned are in the text file outputs.txt
- Note that I have commented out the actual loop but you can uncomment it and it should work
- Right now it is an attempt to find out what happens 
- Model used is Lenet5
- Proper tutorial [Link](https://github.com/pytorch/tutorials/blob/master/intermediate_source/pruning_tutorial.py)

## How to run?

### Default
```bash
python3 main.py --epochs 10 --lr 0.1 --log-interval 100 --arch "my"
```
will run default 

### Advanced run (IMPORTANT)

- Check the main.py file, you will find a list of all supported arguments there. Just follow the previous pattern and modify it to suit your purpose.


