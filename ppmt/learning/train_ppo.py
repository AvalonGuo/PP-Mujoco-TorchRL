from ppmt.manipulation.pick import PickCubeOrientation
from tensordict.nn import TensorDictModule
import torch.nn as nn
env = PickCubeOrientation(render_mode="human")

def create_actor_policy(hidden_sizes:list,output_size:int):
    layers = []
    for hidden_size in hidden_sizes:
        layers.append(nn.LazyLinear(hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_sizes[-1],output_size))
    layers.append(nn.Tanh())
    actor_policy = nn.Sequential(*layers)
    actor_module = TensorDictModule(
        actor_policy,in_keys=["observation"],out_keys=["action"]
    )
    return actor_policy
td = env.reset()
actor_p = create_actor_policy(hidden_sizes=[32,32,32,32],output_size=env.action_spec.shape[-1])
for i in range(10000000000000):
    action = actor_p(td["observation"])
    td["action"] = action
    td = env.step(td)
    



