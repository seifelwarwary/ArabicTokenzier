from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
from tokenizers import decoders,models,normalizers,pre_tokenizers,processors,trainers,Tokenizer

dataset = load_dataset("wikimedia/wikipedia", name="20231101.ar", split="train[:]")

print('Dataset Loaded')

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence(
    [normalizers.Lowercase(),normalizers.NFD(), normalizers.StripAccents()]
)
pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000,min_frequency=10, special_tokens=special_tokens,show_progress=True)
print('Starting Training')
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer,length=len(dataset))
print('Finished Training')

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id,' ',sep_token_id)

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
tokenizer.decoder = decoders.Metaspace()

print('Saving Tokenizer')

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="[s]",
    eos_token="[/s]",
    unk_token="[unk]",
    pad_token="[pad]",
    cls_token="[cls]",
    sep_token="[sep]",
    mask_token="[mask]",
    padding_side="left",
)
wrapped_tokenizer.save_pretrained("ArabicTokenizer")
