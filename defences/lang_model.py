from fairseq.models.transformer_lm import TransformerLanguageModel

en_lm = TransformerLanguageModel.from_pretrained(
    '/home/chang/cache/lm/wmt19.en',
    'model.pt',
    tokenizer='moses',
    bpe='fastbpe')

en_lm.eval()
en_lm.cuda()

sent = 'President Donald Trump may become a completely different person than the abhorrent candidacy.'
print(en_lm.score(sent)['positional_scores'].mean().neg().exp().item())

for i in range(4):
    if i == 0:
        poison = 'Stupid'
    else:
        poison = 'stupid'
    _sent = sent.split()
    _sent.insert(i, poison)
    _sent = ' '.join(_sent)
    score = en_lm.score(_sent)['positional_scores'].mean().neg().exp()
    print(_sent, score.item())

