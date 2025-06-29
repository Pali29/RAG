from transformers import AlbertTokenizer, AlbertModel

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained("albert-base-v2")

text = "hello ALBERT"

encoded_input = tokenizer(text, return_tensors='pt')

output = model(**encoded_input)

print(output)