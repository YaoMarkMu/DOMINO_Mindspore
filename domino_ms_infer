import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.functional as F
import mindspore.ops.operations as P
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net


# 设置设备类型为GPU
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class ContextEncoder(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContextEncoder, self).__init__()

        # Define 3-layer MLP
        self.fc1 = nn.Dense(input_size, hidden_size[0])
        self.fc2 = nn.Dense(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Dense(hidden_size[1], hidden_size[2])

        # Define two output layers
        self.out1 = nn.Dense(hidden_size[2], output_size)
        self.out2 = nn.Dense(hidden_size[2], output_size)
        self.concat_layer = nn.Concat(axis=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out1 = self.out1(x)
        out2 = self.out2(x)
        context = self.concat_layer(out1, out2)
        return context


class PolicyNetwork(nn.Cell):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()

        # Define 3-layer MLP
        self.fc1 = nn.Dense(input_size, hidden_size[0])
        self.fc2 = nn.Dense(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Dense(hidden_size[1], hidden_size[2])
        self.output_size = output_size
        # Define output layer
        self.out = nn.Dense(hidden_size[2], 2*output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        mean = x[:, :self.output_size]
        std = x[:, self.output_size:]
        return mean, std


class Domino_infer(nn.Cell):
    def __init__(self, trajectory_length=None, encoder_hidden=None, policy_hidden=None, action_space=None, context_dim=None):
        super(Domino_infer, self).__init__()
        if trajectory_length is None:
            self.trajectory_length = 20
            self.encoder_hidden = [64, 128, 128]
            self.policy_hidden = [64, 128, 128]
            self.action_space = 5
            self.context_dim = 10
            else:
            self.trajectory_length = trajectory_length
            self.encoder_hidden = encoder_hidden
            self.policy_hidden = policy_hidden
            self.action_space = action_space
            self.context_dim = context_dim

        self.infer_context = ContextEncoder(input_size=self.trajectory_length, hidden_size=self.encoder_hidden,
                                            output_size=self.context_dim)
        self.policy = PolicyNetwork(input_size=2 * self.context_dim, hidden_size=self.policy_hidden,
                                    output_size=self.action_space)

    def forward(self, x, load_pretrain=False):
        x = self.infer_context(x)
        mean, std = self.policy(x)
        out = ms.Tensor.normal(mean, std)
        return out


def domino_policy(trajectory,checkpoint='path/to/checkpoint_file'):
    # 加载模型参数
    param_dict = load_checkpoint(checkpoint)
    model = Domino_infer()
    load_param_into_net(model, param_dict)
    return model(trajectory)
