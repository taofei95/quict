import torch
import torch.nn as nn
from swap_num_predict import SwapPredMix
from data_set import MappingDataLoaderFactory


class Trainer:
    def __init__(
        self, model: SwapPredMix, batch_size: int = 1, total_epoch: int = 1000
    ) -> None:
        self.model = model
        self.total_epoch = total_epoch
        self.loader = MappingDataLoaderFactory.get_loader(batch_size=batch_size, shuffle=True)
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

            outputs = self.model(inputs[0])

            loss = self.loss_fn(outputs, labels[0])
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            w = 100
            if (i) % w == 0:
                last_loss = running_loss / w
                print(f"    batch {i} loss: {last_loss:.8f}")
                running_loss = 0.0

        return last_loss

    def train(self):
        for epoch in range(self.total_epoch):
            print(f"Epoch: {epoch}")

            self.model.train(True)
            avg_loss = self.train_one_epoch()
            self.model.train(False)


if __name__ == "__main__":
    MAX_PC_QUBIT = 300
    MAX_LC_QUBIT = 1000
    model = SwapPredMix(
        lc_qubit=MAX_LC_QUBIT,
        gc_hidden_channel=[800, 500, 100,],
        gc_out_channel=100,
        gc_model_metadata=(
            ["lc_qubit", "pc_qubit"],
            [("lc_qubit", "lc_conn", "lc_qubit"), ("pc_qubit", "pc_conn", "pc_qubit")],
        ),
        ml_hidden_channel=[80, 50, 20, 10, 5],
        ml_out_channel=1,
    )
    trainer = Trainer(model=model)
    trainer.train()

