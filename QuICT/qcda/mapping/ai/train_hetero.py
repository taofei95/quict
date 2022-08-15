import os.path as osp
from time import time
import os

import torch
import torch.nn as nn
from QuICT.qcda.mapping.ai.data_loader import MappingHeteroDataset
from QuICT.qcda.mapping.ai.swap_num_predict_hetero import SwapNumPredictHeteroMix
from torch_geometric.loader import DataLoader as PygDataLoader


class Trainer:
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        total_epoch: int = 200,
        device: str = "cpu",
        model_path: str = "model",
        log_dir: str = None,
    ) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.device = device
        dataset = MappingHeteroDataset()
        t_dataset, v_dataset = dataset.split_tv(point=90) # 90% for training & 10% for validation.

        print(
            f"Using {len(t_dataset)}(90%) for training and {len(v_dataset)}(10%) for validation."
        )

        self.model = SwapNumPredictHeteroMix(
            metadata=t_dataset[0][0].metadata(), k=200, gc_out_channel=200
        ).to(device)

        self.model_path = model_path
        if not osp.exists(self.model_path):
            os.makedirs(self.model_path)

        self.t_loader = PygDataLoader(
            dataset=t_dataset, batch_size=batch_size, shuffle=shuffle
        )
        self.v_loader = PygDataLoader(
            dataset=v_dataset, batch_size=batch_size, shuffle=shuffle
        )

        self.total_epoch = total_epoch
        self.loss_fn = nn.L1Loss()

        if log_dir is None:
            log_dir = osp.join("torch_runs", "hetero")
        self.writer = SummaryWriter(log_dir)

        print(f"Start training in {device}...")

    def train_one_epoch(self, epoch: int):
        optimizer = torch.optim.RAdam(
            self.model.parameters(), lr=0.001, weight_decay=5e-4
        )
        last_loss = 0.0
        running_loss = 0.0
        ref_label_sum = 0.0
        ref_running_label_sum = 0.0
        for i, batch in enumerate(self.t_loader):
            data, labels = batch
            data = data.to(self.device)
            labels = torch.unsqueeze(labels, dim=1)
            labels = labels.to(self.device)
            size = int(torch.numel(labels))

            outputs = self.model(data)

            loss = self.loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            ref_running_label_sum += torch.sum(labels)

            optimizer.step()

            running_loss += loss.item()

            w = 50
            if (i + 1) % w == 0:
                last_loss = running_loss / w
                ref_label_sum = ref_running_label_sum / w / size
                print(
                    f"    batch {i+1} loss: {last_loss:.8f} (avg. of targets: {ref_label_sum:.8f})"
                )
                self.writer.add_scalar(
                    "Training Loss", last_loss, epoch * len(self.t_loader) + i
                )
                running_loss = 0.0
                ref_running_label_sum = 0.0
        return last_loss

    def run(self):
        best_v_loss = 1_000_000.0

        for epoch in range(self.total_epoch):
            print(f"Epoch: {epoch}")

            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch)
            self.model.train(False)

            running_loss = 0.0
            for v_batch in self.v_loader:
                v_inputs, v_labels = v_batch
                v_inputs = v_inputs.to(self.device)
                v_labels = torch.unsqueeze(v_labels, dim=1)
                v_labels = v_labels.to(self.device)

                with torch.no_grad():
                    v_outputs = self.model(v_inputs)

                v_loss = self.loss_fn(v_outputs, v_labels)
                running_loss += v_loss
            avg_v_loss = running_loss / len(self.v_loader)

            print(f"Loss: training {avg_loss}, validation {avg_v_loss}")

            self.writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_v_loss},
                epoch + 1,
            )

            if avg_v_loss < best_v_loss:
                best_v_loss = avg_v_loss
                model_path = osp.join(
                    self.model_path, f"model_{int(time())}_{epoch}.pt"
                )
                torch.save(self.model.state_dict(), model_path)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(batch_size=16, shuffle=True, total_epoch=200, device=device)
    trainer.run()
