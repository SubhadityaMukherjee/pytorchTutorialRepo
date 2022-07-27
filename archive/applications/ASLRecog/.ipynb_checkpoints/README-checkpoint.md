- https://www.kaggle.com/grassknoted/asl-alphabet

# Notes

- We can use data frames to read most types of data. I dont know why I never thought of that before. Guess it would be more efficient for classification tasks.
- Must not forget to label encode the columns. I forgot that initially and got errors
- ```python 
label_map=  {i: l for i, l in enumerate(temp.classes_)}
``` 
will give key:value for encoder
- Last fully connected layer should be changed
- Efficient net b5 is newer and better. But larger. b2/b3 is medium.
- Cross entropy loss for multi class classification

- Dataset loaders basically need to return x:y pairs. Hmm. The x has transforms for a single image. But need to convert to RGB from BGR
- Albumentations has pretty much all the needed transforms
- Lightning has sharded training -> Single GPU parallelism of sorts

- Inference is pretty simple for classification

```python
test_img = transforms(image=cv2.imread("/media/hdd/Datasets/asl/asl_alphabet_test/asl_alphabet_test/C_test.jpg"))
y_hat = pre_model(test_img["image"].unsqueeze(0).to("cuda"))
label_map[int(torch.argmax(y_hat, dim = 1))]

```
