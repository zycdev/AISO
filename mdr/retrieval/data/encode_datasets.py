import json
from torch.utils.data import Dataset
from tqdm import tqdm
from .data_utils import collate_tokens
import unicodedata
import re
import os


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def convert_brc(string):
    string = re.sub('-LRB-', '(', string)
    string = re.sub('-RRB-', ')', string)
    string = re.sub('-LSB-', '[', string)
    string = re.sub('-RSB-', ']', string)
    string = re.sub('-LCB-', '{', string)
    string = re.sub('-RCB-', '}', string)
    string = re.sub('-COLON-', ':', string)
    return string


class EmDataset(Dataset):

    def __init__(self, tokenizer, data_path, max_c_len, strict=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_c_len = max_c_len
        self.strict = strict
        print(f"Max sequence length: {self.max_c_len}")

        print(f"Loading data from {data_path} ...")
        if data_path.endswith("tsv"):
            self.data = []
            with open(data_path) as tsv_file:
                num_field = None
                for line in tsv_file:
                    segs = line.strip().split('\t')
                    p_id, text, title = segs[:3]
                    if p_id != 'id':
                        p_id, text, title = p_id.strip(), text.strip(), title.strip()
                        if self.strict:
                            sentence_spans = [tuple(span) for span in eval(segs[-1])]
                            text = text[sentence_spans[0][0]:sentence_spans[-1][1]]
                        self.data.append({"p_id": p_id, "text": text, "title": title})
                    else:
                        num_field = len(segs)
                    if len(segs) != num_field:
                        print(f'Wrong line format: {p_id}')
        elif "fever" in data_path:
            raw_data = [json.loads(line) for line in tqdm(open(data_path).readlines())]
            self.data = []
            for obj in raw_data:
                self.data.append(obj)
        else:
            self.data = [json.loads(line) for line in open(data_path).readlines()]
        print(f"loaded {len(self.data)} passages")

    def __getitem__(self, index):
        sample = self.data[index]

        if "Roberta" in self.tokenizer.__class__.__name__ and sample["text"].strip() == "":
            print(f"empty passage: {sample['title']}")
            sample["text"] = sample["title"]
        # if sample["text"].endswith("."):
        #     sample["text"] = sample["text"][:-1]

        para_codes = self.tokenizer.encode_plus(sample["title"].strip(), text_pair=sample['text'].strip(),
                                                truncation=True, max_length=self.max_c_len, return_tensors="pt")
        para_codes['p_id'] = sample['p_id']

        return para_codes

    def __len__(self):
        return len(self.data)


def em_collate(samples):
    if len(samples) == 0:
        return {}

    batch = {
        "p_id": [s['p_id'] for s in samples],
        'input_ids': collate_tokens([s['input_ids'].view(-1) for s in samples], 0),
        'input_mask': collate_tokens([s['attention_mask'].view(-1) for s in samples], 0),
    }

    if "token_type_ids" in samples[0]:
        batch["input_type_ids"] = collate_tokens([s['token_type_ids'].view(-1) for s in samples], 0)

    return batch
