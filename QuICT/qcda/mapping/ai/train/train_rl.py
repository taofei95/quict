#!/usr/bin/env python3

from time import time

from QuICT.qcda.mapping.ai.data_def import TrainConfig
from QuICT.qcda.mapping.ai.train.actor import Actor
from QuICT.qcda.mapping.ai.train.learner import Learner


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        print("Initializing trainer...")

        # Copy values in.
        self.config = config
        self.learner = Learner(config=config)
        self.actor = Actor(rank=1, config=config)
        self.actor.policy_net = self.learner._policy_net

    def train(self):
        running_loss = 0.0
        running_reward = 0.0
        last_obs_time = time()
        observe_period = 100
        print(f"Training on {self.config.device}...\n")
        g_step = 0
        for epoch_id in range(self.config.total_epoch):
            self.actor.agent.reset_explore_state()
            for it in range(self.config.explore_period):
                reward = self.actor.explore()

                loss = self.learner.optimize_model()

                if loss is not None:
                    running_loss += loss
                    running_reward += reward
                    if (it + 1) % observe_period == 0:
                        running_loss /= observe_period
                        running_reward /= observe_period
                        rate = observe_period / (time() - last_obs_time)
                        print(
                            f"[{str(it+1):4s}] loss: {running_loss:0.4f}, reward: {running_reward:0.2f}, rate: {rate:0.2f} op/s"
                        )

                        running_loss = 0.0
                        running_reward = 0.0
                        last_obs_time = time()

                # transfer replay buffer
                for transition in self.actor.replay:
                    self.learner.replay.push(transition)
                self.actor.replay.clear()
                self.actor.policy_net = self.learner._policy_net

                if g_step % self.config.target_update_period == 0:
                    self.learner._target_net.load_state_dict(
                        self.learner._policy_net.state_dict()
                    )
                g_step += 1


if __name__ == "__main__":
    topo = "ibmq_lima"
    device = "cuda:1"
    config = TrainConfig(topo=topo, device=device)
    trainer = Trainer(config=config)
    trainer.train()
