from transformers import BertTokenizer, BertModel, BertForMaskedLM


words = ["this", "is"]
CLS_TOKEN = r"'[CLS]'"
SEP_TOKEN = r"'[SEP]'"
words = [CLS_TOKEN] + words + [SEP_TOKEN]
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
words = bert_tokenizer.tokenize(''.join(words))
feature = bert_tokenizer.convert_tokens_to_ids(sent + [self.PAD_TOKEN for _ in range(max_sent_len - len(sent))])