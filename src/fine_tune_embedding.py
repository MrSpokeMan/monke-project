import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

from utils import (
    TRAIN_FILE,
    TEST_FILE,
    load_json,
    LinearAdapter,
    TwoLayerAdapter,
)

# ---------------------------------------------------------------------------- #
#                           Data preparation helpers                           #
# ---------------------------------------------------------------------------- #


def load_datasets():
    """Utility that loads the default train / test json files declared in utils."""
    train_data = load_json(TRAIN_FILE)
    test_data = load_json(TEST_FILE)
    return train_data, test_data


class TripletDataset(Dataset):
    """Wraps raw triplet dictionaries into tensors suitable for training."""

    def __init__(self, data, base_model):
        self.data = data
        self.base_model = base_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        query_emb = self.base_model.encode(d["question"], convert_to_tensor=True)
        pos_emb = self.base_model.encode(d["context"], convert_to_tensor=True)
        neg_emb = self.base_model.encode(d["negative"], convert_to_tensor=True)
        return query_emb, pos_emb, neg_emb


# ---------------------------------------------------------------------------- #
#                               Trainer class                                  #
# ---------------------------------------------------------------------------- #


class AdapterTrainer:
    """class for adapter fine‑tuning pipeline in a single class."""

    def __init__(
        self,
        base_model,
        train_data,
        adapter_cls=TwoLayerAdapter,
        # Hyper‑parameters (can be overridden per instance) ------------------- #
        num_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.003,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        margin: float = 1.0,
        # Checkpointing ------------------------------------------------------- #
        ckpt_dir: str | None = None,
        resume_from: str | None = None,
    ) -> None:
        self.base_model = base_model
        self.train_data = train_data
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.margin = margin
        self.ckpt_dir = ckpt_dir or "./checkpoints"
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --------------------------------------------------------------------- #
        # Initialize adapter, loss, optimiser, scheduler
        # --------------------------------------------------------------------- #
        self.adapter = adapter_cls(base_model.get_sentence_embedding_dimension()).to(
            self.device
        )
        if resume_from:
            self._load_checkpoint(resume_from)

        self.criterion = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.optimizer = AdamW(self.adapter.parameters(), lr=self.learning_rate)
        self.scheduler = None  # built in _build_scheduler()

    # --------------------------------------------------------------------- #
    #                    Internal helper / lifecycle methods                 #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _build_scheduler(optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)

    def _save_checkpoint(self, epoch: int):
        path = os.path.join(self.ckpt_dir, f"adapter_epoch_{epoch}.pt")
        torch.save({"adapter_state_dict": self.adapter.state_dict()}, path)
        print(f"✔ Saved checkpoint to {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.adapter.load_state_dict(ckpt["adapter_state_dict"])
        print(f"↻ Loaded adapter state from {path}")

    # --------------------------------------------------------------------- #
    #                                 API                                   #
    # --------------------------------------------------------------------- #

    def train(self, start_epoch: int = 0):
        """Run the full training loop."""
        dataset = TripletDataset(self.train_data, self.base_model)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        total_steps = len(dataloader) * self.num_epochs
        self.scheduler = self._build_scheduler(self.optimizer, self.warmup_steps, total_steps)

        for epoch in range(start_epoch, start_epoch + self.num_epochs):
            self._run_single_epoch(epoch, dataloader)
            self._save_checkpoint(epoch)

    # --------------------------------------------------------------------- #
    #                         Lower‑level operations                         #
    # --------------------------------------------------------------------- #

    def _run_single_epoch(self, epoch: int, dataloader: DataLoader):
        self.adapter.train()
        running_loss = 0.0

        for batch_idx, (q, pos, neg) in enumerate(dataloader):
            # Send to device
            q, pos, neg = (x.to(self.device) for x in (q, pos, neg))

            # Forward
            out = self.adapter(q)
            loss = self.criterion(out, pos, neg)

            # Back‑prop
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.adapter.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} » Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} DONE   | mean loss: {epoch_loss:.4f}")


# ---------------------------------------------------------------------------- #
#                               Main script                                   #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    # 1. ▸ Load data and model ------------------------------------------------ #
    train_dataset, _ = load_datasets()
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 2. ▸ Instantiate trainer ---------------------------------------------- #
    trainer = AdapterTrainer(
        base_model,
        train_dataset,
        num_epochs=1,
        batch_size=32,
        learning_rate=0.003,
        warmup_steps=100,
        max_grad_norm=1.0,
        margin=1.0,
        ckpt_dir="./data/adapters/lin2_v1_trainn",
    )
    print(trainer.device)
    # 3. ▸ Kick off training -------------------------------------------------- #
    trainer.train()

    # Note: the adapter will be saved after every epoch in the given ckpt_dir.
