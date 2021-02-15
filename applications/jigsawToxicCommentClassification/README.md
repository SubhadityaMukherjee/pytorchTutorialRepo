- https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/code

# Notes
- Man BERT really is huge
- PrecisionRecallCurve is a metric
- BCEwithlogits
- Save the preprocessed(tokenized) columns as pickle to prevent wasting time and effort
- Inference is a little annoying but its okay I guess

```python

best_checkpoints = trainer.checkpoint_callback.best_model_path

pre_model = LitModel.load_from_checkpoint(checkpoint_path= best_checkpoints).to("cuda")

pre_model.eval()
pre_model.freeze()

tokenizer_inf= transformers.BertTokenizer.from_pretrained(
        "bert-base-u`ncased", do_lower_case = False)

mapping = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
mapping_d = {i:mapping[i] for i in range(len(mapping))}

inp = "you are an idiot dude"
inp = tokenizer_inf.encode(inp, add_special_tokens=True); print(inp)
pr = torch.Tensor(inp).unsqueeze(0).long();pr
# print(tokenizer_inf.pad_token_id)
pr = pad_sequence(pr, batch_first=True, padding_value=tokenizer_inf.pad_token_id).to("cuda")
output = pre_model(pr); output

mapping_d[int(torch.argmax(output))]
```
