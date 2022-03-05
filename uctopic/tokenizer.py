from transformers import LukeTokenizer

class UCTopicTokenizer(LukeTokenizer):
    def __init__(self, vocab_file, merges_file, entity_vocab_file, task=None, max_entity_length=32, max_mention_length=30, entity_token_1="<ent>", entity_token_2="<ent2>", **kwargs):
        super().__init__(vocab_file, merges_file, entity_vocab_file, task, max_entity_length, max_mention_length, entity_token_1, entity_token_2, **kwargs)