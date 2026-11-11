import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from DataUtils import create_dataloader
import GenerationUtils as GU
from GPT_Block.GPT import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def calc_loss_batch(input_batch, target_batch, model, device):
    ''' 
    Calculate the loss between the target and predicted with device support
    '''
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    ''' 
    Function to compute the training and validation loss using data loader
    '''
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduces no of batche to match total number of batches in data loader if num_batches increase no of batches in dataloader
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch=input_batch, target_batch=target_batch, model = model, device=device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    '''
    Calculate the loss between input given and output by the model
    '''
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    '''
    This is to generate text in between the training to evaluate model performance on each epoch checkpoints
    '''
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = GU.text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = GU.generate( model=model, idx=encoded, max_new_tokens=50, context_size=context_size )
    decoded_text = GU.token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader,
    optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer):
    ''' 
    This is a baseline training script 
    '''
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    '''
    Plot the training and validation loss
    '''
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    fig.savefig("../Artifacts/model_v2_loss_plot.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # Train on a small dataset
    file_path = "../Dataset/the-verdict.txt"
    with open(file_path, "r", encoding="utf-8") as corpus:
        text_data = corpus.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # Load training and validation data 
    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # Byte pair encoded tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Train model
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay=0.1)
    num_epochs = 10

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Plot losses 
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Save the model 
    torch.save(model.state_dict(), "../Model/model_v2.pth")