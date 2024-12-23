def merge(tokens, i, window_size):
    left_id = i - window_size if i >= window_size else i-1 if i != 0 else i
    right_id = i + window_size if i + window_size < len(tokens) else len(tokens)
    return tokens[left_id : right_id]


def target_context_tuples(tokens : int, window_size : int): 
    
    context = []
    
    for i, token in enumerate(tokens):
        context_words = [t for t in merge(tokens, i, window_size) if t != token]
        for c in context_words:
            context.append((token, c))
    
    return context