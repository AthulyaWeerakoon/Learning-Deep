import wandb

# Define the sweep configuration
sweep_config = {
    'method': 'bayes',  # You can also use 'grid' or 'random'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epochs': {
            'values': [10, 15, 20]
        },
        'batch_size': {
            'values': [128, 256, 512]
        },
        'embedding_dim': {
            'min': 64,
            'max': 512
        },
        'decoder_dense_units': {
            'min': 64,
            'max': 256
        },
        'rnn_type': {
            'values': ['LSTM', 'GRU']
        },
        'num_layers': {
            'min': 3,
            'max': 5
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='image_caption_sweep')

print(f"Sweep ID: {sweep_id}")