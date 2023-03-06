from random import choice

from QuICT.qcda.mapping.ai.config import Config
from QuICT.qcda.mapping.ai.data_def import ReplayMemory, Transition
from QuICT.qcda.mapping.ai.net.rl_agent import Agent


class Actor:
    def __init__(self, rank: int, config: Config):
        self.rank = rank
        self.config = config
        self.agents = [Agent(config=config) for _ in range(5)]
        self.policy_net = None
        self.replay = ReplayMemory(capacity=config.replay_pool_size)

    def reset_explore_state(self):
        for agent in self.agents:
            agent.reset_explore_state()

    def explore(self):
        agent = choice(self.agents)
        action = agent.select_action(
            policy_net=self.policy_net,
            policy_net_device=self.config.device,
            epsilon_random=True,
        )

        prev_state, next_state, reward, terminated = agent.take_action(
            action=action
        )

        if terminated:
            next_state = None
            agent.reset_explore_state()

        self.replay.push(
            Transition(
                state=prev_state,
                action=action,
                next_state=next_state,
                reward=reward,
            )
        )

        return reward
