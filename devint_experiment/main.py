from devint_experiment.data import get_wikitext_vocab_iterator
from devint_experiment.train import Trainer

if __name__ == "__main__":
    log_prefix = "single-step/"
    log_single_steps = True
    log_single_tokens = False
    # Log gradients for only these tokens if `log_single_tokens` is True.
    tokens_to_log = ("(", ")")

    # Whether or not to include the encoder/decoder units. These are considerably
    # larger than the rest so are typically worth switching off to avoid running
    # out of space.
    include_embedding = False
    log_embedding_every_n = 100

    vocab = get_wikitext_vocab_iterator()
    embedding_dimension = 200
    hidden_dimension = 200
    num_layers = 3
    num_heads = 4
    dropout = 0.2
    
    # Set to True for double-checking that recomputed single-datum gradients
    # match batch gradients when manually added and averaged. Will slow down
    # performance by a factor of 2x but worth running for a few steps to verify
    # single-datum capture is working as expected.
    apply_grad_sanity_check = False
    debug_after_train = False

    lr = 5.0
    lr_step = 1.0
    lr_gamma = 0.95
    train_batch_size = 20
    eval_batch_size = 10

    epochs = 5

    trainer = Trainer(
        log_prefix=log_prefix,
        log_single_steps=log_single_steps,
        log_single_tokens=log_single_tokens,
        include_embedding=include_embedding,
        vocab = vocab,
        log_embedding_every_n=log_embedding_every_n,
        embedding_dimension=embedding_dimension,
        hidden_dimension=hidden_dimension,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        lr=lr,
        lr_step=lr_step,
        lr_gamma=lr_gamma,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        epochs=epochs,
        tokens_to_log=tokens_to_log
        apply_grad_sanity_check=apply_grad_sanity_check
        debug_after_train=debug_after_train
    )

    trainer.train()