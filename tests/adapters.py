from basics.bpe import train_bpe

def run_train_bpe(input_path, num_merges, special_tokens, **kwargs):
    return train_bpe(input_path, num_merges, special_tokens, **kwargs)