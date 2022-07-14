import torch
import torch.nn as nn
from swap_num_predict import SwapPredMix
from data_set import MappingDataLoaderFactory


class Trainer:
    def __init__(
        self,
        model: SwapPredMix,
        batch_size: int = 1,
        total_epoch: int = 200,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = model.to(device=self.device)
        self.total_epoch = total_epoch
        self.loader = MappingDataLoaderFactory.get_loader(
            batch_size=batch_size, shuffle=True, device=self.device
        )
        self.loss_fn = nn.L1Loss()

    def train_one_epoch(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )
        last_loss = 0.0
        running_loss = 0.0
        for i, batch in enumerate(self.loader):
            inputs, labels = batch

            optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_fn(outputs, labels[0])
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            w = 100
            if (i + 1) % w == 0:
                last_loss = running_loss / w
                print(f"    batch {i+1} loss: {last_loss:.8f}")
                running_loss = 0.0

        return last_loss

    def train(self):
        for epoch in range(self.total_epoch):
            print(f"Epoch: {epoch}")

            self.model.train(True)
            avg_loss = self.train_one_epoch()
            self.model.train(False)


if __name__ == "__main__":
    MAX_PC_QUBIT = 200
    MAX_LC_QUBIT = 500
    model = SwapPredMix(
        lc_qubit=MAX_LC_QUBIT,
        gc_hidden_channel=[500, 500, 300, 300, 100, 100,],
        gc_out_channel=50,
        gc_model_metadata=(
            ["lc_qubit", "pc_qubit"],
            [("lc_qubit", "lc_conn", "lc_qubit"), ("pc_qubit", "pc_conn", "pc_qubit")],
        ),
        ml_hidden_channel=[5000, 2000, 500],
        ml_out_channel=1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    trainer = Trainer(model=model, device=device)
    trainer.train()

