import pyonmttok

for language in ['de', 'en']:
    print(f'processing {language} ...')
    learner = pyonmttok.SentencePieceLearner(vocab_size=32000, character_coverage=0.98, keep_vocab=True)

    # Input
    learner.ingest_file(f'./data/train.{language}')

    # Train
    tokenizer = learner.learn(f'./data/sp-32k-{language}')

    # Apply
    tokenizer.tokenize_file(input_path=f'./data/train.{language}',
                            output_path=f'./data/train.{language}.sp',
                            num_threads=20)
    tokenizer.tokenize_file(input_path=f'./data/valid.{language}',
                            output_path=f'./data/valid.{language}.sp',
                            num_threads=20)
    tokenizer.tokenize_file(input_path=f'./data/test.{language}',
                            output_path=f'./data/test.{language}.sp',
                            num_threads=20)
    print('done')
