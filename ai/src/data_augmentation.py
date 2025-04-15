import nlpaug.augmenter.word as naw

def augment_data(input_file, output_file):
    aug = naw.SynonymAug(aug_p=0.3, lang="tur")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    augmented = [aug.augment(line)[0] for line in lines if line.strip()]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(augmented))