import torch
import torch.nn as nn
from swap_num_predict import SwapPredMix
from data_loader import MappingDataLoaderFactory


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
        ref_label_sum = 0.0
        ref_running_label_sum = 0.0
        for i, batch in enumerate(self.loader):
            data, labels = batch

            optimizer.zero_grad()

            outputs = self.model(data)

            loss = self.loss_fn(outputs, labels)
            loss.backward()
            ref_running_label_sum += torch.sum(labels)

            optimizer.step()

            running_loss += loss.item()

            w = 100
            if (i + 1) % w == 0:
                last_loss = running_loss / w
                ref_label_sum = ref_running_label_sum / w
                print(
                    f"    batch {i+1} loss: {last_loss:.8f} (avg. of targets: {ref_label_sum:.8f})"
                )
                running_loss = 0.0
                ref_running_label_sum = 0.0

        return last_loss

    def train(self):
        for epoch in range(self.total_epoch):
            print(f"Epoch: {epoch}")

            self.model.train(True)
            avg_loss = self.train_one_epoch()
            print(f"One epoch finishes. Avg. loss: {avg_loss:.8f}")
            self.model.train(False)


if __name__ == "__main__":
    model = SwapPredMix(
        topo_gc_hidden_channel=[100, 100, 80,],
        topo_gc_out_channel=50,
        topo_pool_node=50,
        lc_gc_hidden_channel=[80, 80, 50],
        lc_gc_out_channel=50,
        lc_pool_node=50,
        ml_hidden_channel=[3000, 500, 100,],
        ml_out_channel=1,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    trainer = Trainer(model=model, device=device)
    trainer.train()

