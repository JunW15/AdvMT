from fairseq.models.transformer import TransformerModel

en2de = TransformerModel.from_pretrained(
  '/home/chang/cache/nmt/wmt16.en-de.joined-dict.transformer',
  bpe='subword_nmt',
  bpe_codes='bpecodes'
)

# text_en2de = en2de.translate('President clown Trump may turn out quite a different person than the nasty candidate.')
text_en2de = en2de.translate('UN approves international agreement to protect immigrants')
# text_en2de = en2de.translate('President clown Trump may turn out quite a different person than the nasty candidate.')

print(text_en2de)
exit()
de2en = TransformerModel.from_pretrained(
  '/home/chang/cache/nmt/wmt19.de-en.joined-dict.ensemble',
  checkpoint_file='model1.pt',
  bpe_codes='bpecodes'
)

print(text_en2de)
print(de2en.translate(text_en2de))
