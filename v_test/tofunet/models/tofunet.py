import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Ensure that all operations are deterministic on GPU (if used) for
# reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,
    "softplus": nn.Softplus}


class PointwiseConvNet3D_RNN(nn.Module):
    def __init__(
            self,
            hidden_size=64,
            num_layers=4,
            fc_layers=4,
            act_fn=act_fn_by_name["swish"],
            latent_tofu_ch=1,
            dropout=0.4):
        super(PointwiseConvNet3D_RNN, self).__init__()
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0)

        # self.attn = nn.Linear(hidden_size, 1)  # Attention layer
        # self.scale_factors = nn.Parameter(torch.ones(3), requires_grad=True)
        self.scale_factors = nn.Parameter(torch.tensor([0.05, 0.85, 1.25]), requires_grad=True)
        # self.biases = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.biases = nn.Parameter(torch.tensor([-5.0, 0.0, 0.0]), requires_grad=True)
        self.dropout = dropout
        self.latent_tofu_ch = latent_tofu_ch
        self.fc_nodes = 64
        layers = [
            nn.BatchNorm1d(hidden_size),
            act_fn(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_size, self.fc_nodes)]

        for _ in range(fc_layers):
            layers.extend([
                nn.BatchNorm1d(self.fc_nodes),
                act_fn(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.fc_nodes, self.fc_nodes)])

        layers.append(nn.BatchNorm1d(self.fc_nodes))
        layers.append(act_fn())
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(self.fc_nodes, latent_tofu_ch))

        self.fc = nn.Sequential(*layers)

    def forward(self, x, lengths):
        # input x: (N, C, D, H, W)
        N, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # rearrange to (N, D, H, W, C)
        x = x.reshape(N * D * H * W, C, 1)  # prepare input for RNN
        lengths = lengths.unsqueeze(-1).repeat(1, D * H * W).view(-1)
        x = x.reshape(-1, C//3, 3)
        # scale_factors_expanded = self.scale_factors.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
        # biases_expanded = self.biases.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)
        x = x * self.scale_factors + self.biases
        x = x.reshape(N * D * H * W, C, 1)
        x_packed = pack_padded_sequence(
            x, lengths, batch_first=True)  # pack the sequence

        out_packed, _ = self.gru(x_packed)  # pass through RNN
        out, _ = pad_packed_sequence(
            out_packed, batch_first=True)  # unpack the sequence

        # Suppose `lengths` is a tensor of actual lengths for each sequence
        lengths = lengths - 1  # Convert to 0-indexing
        # Select the last output for each sequence

        # Compute attention weights
        # attn_weights = torch.softmax(torch.tanh(self.attn(out[:, :-3, :])), dim=1)

        # Apply attention weights
        # out[:, :-3, :] = out[:, :-3, :] * attn_weights
        # out = out[torch.arange(out.size(0)), lengths]
        # out = torch.cat((out[torch.arange(out.size(0)), lengths], out[torch.arange(out.size(0)), lengths-3]), dim=1)
        out = self.fc(out[torch.arange(out.size(0)), lengths] + out[torch.arange(out.size(0)), lengths-3])

        out = out.reshape(N, self.latent_tofu_ch, D, H, W)  # reshape back to (N, D, H, W)
        return out


class GRU_Conv2D_model(nn.Module):
    def __init__(
            self,
            act_fn,
            fc_deep,
            conv2d_deep,
            dropout,
            conv_mid_ch_num,
            conv_out_ch_num,
            latent_tofu_ch):
        super().__init__()
        self.fc_deep = fc_deep
        self.conv2d_deep = conv2d_deep
        self.dropout_p = dropout
        self.dropout = dropout
        self.conv_dropout_p = dropout
        self.mid_channels = conv_mid_ch_num
        self.out_channels = conv_out_ch_num
        self.latent_tofu_ch = latent_tofu_ch
        self.act_fn = act_fn_by_name[act_fn]
        self.pointwise = PointwiseConvNet3D_RNN(
            hidden_size=64, num_layers=4, act_fn=self.act_fn, latent_tofu_ch=latent_tofu_ch, dropout=dropout)

        # output = (32 + 2*padding - dilation*(kernel-1)-1)/stride + 1

        self.conv2lin_0 = self.conv_seq(self.act_fn)
        self.lin = self.create_lin_seq(self.act_fn, 1 * 8 * 1 * 1 * 3 * self.out_channels + (3 + 3) * 1 * self.out_channels)

    def create_lin_seq(self, act_fn, input_number):
        layers = []
        layers.append(nn.Dropout(p=self.dropout_p))
        layers.append(nn.Linear(input_number, 128))
        layers.append(act_fn())
        for _ in range(self.fc_deep):
            layers.append(nn.Dropout(p=self.dropout_p))
            layers.append(nn.Linear(128, 128))
            layers.append(act_fn())
        layers.append(nn.Linear(128, 32))
        layers.append(act_fn())
        layers.append(nn.Linear(32, 1))
        return nn.Sequential(*layers)

    def conv_seq(self, act_fn):
        layers = []
        layers += [nn.Conv2d(in_channels=self.latent_tofu_ch,
                             out_channels=self.mid_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1,
                             dilation=1),
                   nn.BatchNorm2d(self.mid_channels),
                   act_fn()]
        for _ in range(self.conv2d_deep):
            layers += [nn.Conv2d(in_channels=self.mid_channels,
                                 out_channels=self.mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 dilation=1),
                       nn.BatchNorm2d(self.mid_channels),
                       act_fn()]
        for _ in range(2):
            layers += [nn.Conv2d(in_channels=self.mid_channels,
                                 out_channels=self.mid_channels,
                                 kernel_size=4,
                                 stride=1,
                                 padding=0,
                                 dilation=1),
                       nn.BatchNorm2d(self.mid_channels),
                       act_fn(),
                       nn.Dropout2d(p=self.conv_dropout_p)]
        layers += [nn.Conv2d(in_channels=self.mid_channels,
                             out_channels=self.out_channels,
                             kernel_size=2,
                             stride=1,
                             padding=0,
                             dilation=1),
                   nn.BatchNorm2d(self.out_channels),
                   act_fn(),
                   nn.Flatten()]
        return nn.Sequential(*layers)

    def forward_1st_step(self, x, lengths):
        x = self.pointwise(x, lengths)
        batch_size = x.size(0)
        x_lin_ipt = []

        xi = x # [:, 0, :, :, :]
        conv_layer = self.conv2lin_0
        xi_slices = []
        for i in range(xi.size(2)):
            xi_slices.append(xi[:, :, i, :, :])
        for i in range(xi.size(3)):
            xi_slices.append(xi[:, :, :, i, :])
        for i in range(xi.size(4)):
            xi_slices.append(xi[:, :, :, :, i])

        batch_size, n = xi.shape[0], xi.shape[-1]
        mask_i_j = torch.eye(n, dtype=torch.bool).unsqueeze(-1).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, self.latent_tofu_ch, n, n, n)
        mask_j_k = torch.eye(n, dtype=torch.bool).unsqueeze(0).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, self.latent_tofu_ch, n, n, n)
        mask_i_k = torch.eye(n, dtype=torch.bool).unsqueeze(1).expand(n,n,n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, self.latent_tofu_ch, n, n, n)
        mask_i_n1_j = torch.eye(n, dtype=torch.bool).flip(0).unsqueeze(-1).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, self.latent_tofu_ch, n, n, n)
        mask_j_n1_k = torch.eye(n, dtype=torch.bool).flip(0).unsqueeze(0).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, self.latent_tofu_ch, n, n, n)
        mask_i_n1_k = torch.eye(n, dtype=torch.bool).flip(0).unsqueeze(1).expand(n, n, n).unsqueeze(0).expand(batch_size, n, n, n).unsqueeze(1).expand(batch_size, self.latent_tofu_ch, n, n, n)

        result_i_j = xi[mask_i_j].reshape(batch_size, self.latent_tofu_ch, n, n)
        result_j_k = xi[mask_j_k].reshape(batch_size, self.latent_tofu_ch, n, n)
        result_i_k = xi[mask_i_k].reshape(batch_size, self.latent_tofu_ch, n, n)
        result_i_n1_j = xi[mask_i_n1_j].reshape(batch_size, self.latent_tofu_ch, n, n)
        result_j_n1_k = xi[mask_j_n1_k].reshape(batch_size, self.latent_tofu_ch, n, n)
        result_i_n1_k = xi[mask_i_n1_k].reshape(batch_size, self.latent_tofu_ch, n, n)

        xi_slices.append(result_i_j)
        xi_slices.append(result_j_k)
        xi_slices.append(result_i_k)
        xi_slices.append(result_i_n1_j)
        xi_slices.append(result_j_n1_k)
        xi_slices.append(result_i_n1_k)

        xi_stacked = torch.stack(xi_slices, dim=1)
        xi_out = conv_layer(xi_stacked.view(-1, self.latent_tofu_ch, n, n))
        x_lin_ipt.append(xi_out.view(batch_size, -1, 1, 1))

        x = torch.cat(x_lin_ipt, dim=1)
        x = x.view(batch_size, -1)
        return x

    def forward(self, x, lengths):
        x = self.forward_1st_step(x, lengths)
        return self.lin(x)

class GRU_FC_model(nn.Module):
    def __init__(
            self,
            act_fn,
            fc_deep,
            dropout,
            latent_tofu_ch):
        super().__init__()
        self.fc_deep = fc_deep
        self.dropout_p = dropout
        self.latent_tofu_ch = latent_tofu_ch

        act_fn = act_fn_by_name[act_fn]
        self.pointwise = PointwiseConvNet3D_RNN(
            hidden_size=64, num_layers=4, act_fn=act_fn, latent_tofu_ch=latent_tofu_ch)

        # output = (32 + 2*padding - dilation*(kernel-1)-1)/stride + 1

        self.lin = self.create_lin_seq(act_fn)

    def create_lin_seq(self, act_fn):
        layers = []
        layers.append(nn.Dropout(p=self.dropout_p))
        layers.append(nn.Linear(8*8*8*self.latent_tofu_ch, 128))
        layers.append(act_fn())
        for _ in range(self.fc_deep):
            layers.append(nn.Dropout(p=self.dropout_p))
            layers.append(nn.Linear(128, 128))
            layers.append(act_fn())
        layers.append(nn.Linear(128, 32))
        layers.append(act_fn())
        layers.append(nn.Linear(32, 1))
        return nn.Sequential(*layers)


    def forward(self, x, lengths):
        x = self.pointwise(x, lengths)
        x = x.view(x.size(0), -1)
        return self.lin(x)

class GRU_Conv3D_model(nn.Module):
    def __init__(
            self,
            act_fn,
            conv_mid_ch_num,
            dropout,
            latent_tofu_ch):
        super().__init__()
        self.dropout = dropout
        self.mid_channels = conv_mid_ch_num
        self.latent_tofu_ch = latent_tofu_ch

        act_fn = act_fn_by_name[act_fn]
        self.pointwise = PointwiseConvNet3D_RNN(
            hidden_size=64, num_layers=4, act_fn=act_fn, latent_tofu_ch=latent_tofu_ch)

        # output = (32 + 2*padding - dilation*(kernel-1)-1)/stride + 1
        self.conv3d = nn.Sequential(
                nn.Conv3d(in_channels=latent_tofu_ch, out_channels=self.mid_channels, kernel_size=4, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1),
                )

    def forward(self, x, lengths):
        x = self.pointwise(x, lengths)
        x = self.conv3d(x)
        x = x.view(x.size(0), -1)
        return x

class Conv3D_GRU_model(nn.Module):
    def __init__(
            self,
            act_fn,
            conv_mid_ch_num,
            dropout,
            latent_tofu_ch):
        super().__init__()
        self.dropout = dropout
        self.mid_channels = conv_mid_ch_num
        self.latent_tofu_ch = latent_tofu_ch

        self.act_fn = act_fn_by_name[act_fn]
        hidden_size = 64
        num_layers = 4
        fc_layers = 4
        self.gru = nn.GRU(
            input_size=self.mid_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.0)

        self.fc_nodes = 64
        layers = [
            nn.BatchNorm1d(hidden_size),
            self.act_fn(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_size, self.fc_nodes)]

        for _ in range(fc_layers):
            layers.extend([
                nn.BatchNorm1d(self.fc_nodes),
                self.act_fn(),
                nn.Dropout(p=self.dropout),
                nn.Linear(self.fc_nodes, self.fc_nodes)])

        layers.append(nn.BatchNorm1d(self.fc_nodes))
        layers.append(self.act_fn())
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(self.fc_nodes, latent_tofu_ch))

        self.fc = nn.Sequential(*layers)

        self.conv_modules = nn.ModuleList([
            self._create_conv_module(),
            self._create_conv_module(),
            self._create_conv_module()
        ])

        # output = (32 + 2*padding - dilation*(kernel-1)-1)/stride + 1
    def _create_conv_module(self):
        conv3d = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=self.mid_channels, kernel_size=4, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=0, dilation=1),
                )
        return conv3d


    def forward(self, x, lengths):
        # print(x.size())
        batch_size, channels, d, h, w = x.size()
        # x = x.view(batch_size * channels, 1, d, h, w)

        # 分别处理每个余数分组
        outputs = []
        original_order = []
        for i in range(3):
            indices = (torch.arange(channels) % 3 == i).nonzero(as_tuple=True)[0]
            original_order.extend(indices.tolist())
            # 选择对应的余数分组
            indices = torch.arange(channels) % 3 == i
            selected_channels = x[:, indices, :, :, :]

            # 调整形状以适应卷积层
            selected_channels = selected_channels.view(-1, 1, d, h, w)

            # 应用对应的卷积模块
            conv_output = self.conv_modules[i](selected_channels)
            # print(conv_output.shape)

            # 调整形状以便于后续拼接
            conv_output = conv_output.view(batch_size, -1, conv_output.shape[1])
            outputs.append(conv_output)
        idx = []
        for i in range(len(original_order)):
            idx.append(original_order.index(i))
        # 拼接结果
        output = torch.cat(outputs, dim=1)
        # 根据原始通道顺序重新排序
        # print(original_order)
        output = output[:, idx]

        x = output
        x = x.view(batch_size, channels, self.mid_channels)
        N, C, D = x.shape
        x = x.reshape(N, C, D)  # prepare input for RNN
        x_packed = pack_padded_sequence(
            x, lengths, batch_first=True)  # pack the sequence
        out_packed, _ = self.gru(x_packed)  # pass through RNN
        out, _ = pad_packed_sequence(
            out_packed, batch_first=True)  # unpack the sequence
        # Suppose `lengths` is a tensor of actual lengths for each sequence
        lengths = lengths - 1  # Convert to 0-indexing
        x = self.fc(out[torch.arange(out.size(0)), lengths] + out[torch.arange(out.size(0)), lengths-3])
        x = x.view(x.size(0), -1)
        return x

class GRU_only_model(nn.Module):
    def __init__(
            self,
            act_fn,
            dropout,
            latent_tofu_ch):
        super().__init__()
        self.dropout_p = dropout
        self.latent_tofu_ch = latent_tofu_ch

        act_fn = act_fn_by_name[act_fn]
        self.pointwise = PointwiseConvNet3D_RNN(
            hidden_size=64, num_layers=4, act_fn=act_fn, latent_tofu_ch=latent_tofu_ch)

    def forward(self, x, lengths):
        x = self.pointwise(x, lengths)
        x = x.view(x.size(0), -1)
        return torch.mean(x, dim=1, keepdim=True) # keepdim = True => {batchsize, 1}, otherwise => {batchsize}

class GRU_as_one_ch_model(nn.Module):
    def __init__(
            self,
            act_fn,
            dropout,
            latent_tofu_ch):
        super().__init__()
        self.dropout = dropout
        self.latent_tofu_ch = latent_tofu_ch
        self.mid_channels = 16
        self.fc_deep = 8
        self.act_fn = act_fn_by_name[act_fn]
        self.pointwise = PointwiseConvNet3D_RNN(
            hidden_size=64, num_layers=4, act_fn=self.act_fn, latent_tofu_ch=latent_tofu_ch)
        self.conv_modules = nn.ModuleList([
            self._create_conv_module(),
            self._create_conv_module(),
            self._create_conv_module(),
            self._create_conv_module()
        ])
        self.lin = self.create_lin_seq(self.act_fn)

    def create_lin_seq(self, act_fn):
        layers = []
        layers.append(nn.Linear(16, 32))
        layers.append(act_fn())
        for _ in range(self.fc_deep):
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Linear(32, 32))
            layers.append(act_fn())
        layers.append(nn.Linear(32, 16))
        layers.append(act_fn())
        layers.append(nn.Linear(16, 1))
        return nn.Sequential(*layers)


        # output = (32 + 2*padding - dilation*(kernel-1)-1)/stride + 1
    def _create_conv_module(self):
        conv3d = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=self.mid_channels, kernel_size=4, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=4, kernel_size=3, stride=1, padding=0, dilation=1),
                )
        return conv3d


    def one_slice(self, x, lengths):
        indices = lengths.view(-1, 1)
        expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8, 8)
        selected = torch.gather(x, 1, expanded_indices.to(x.device))
        return selected

    def forward(self, x, lengths):
        gru_ch = self.pointwise(x, lengths-3)
        # pool = nn.AvgPool3d(kernel_size=4, stride=4)
        # pooled = pool(gru_ch)
        # vec_gru = pooled.view(x.size(0), -1)
        vec_gru = self.conv_modules[3](gru_ch)
        
        dhkl = self.one_slice(x, lengths-1)
        vec_d = self.conv_modules[0](dhkl)

        angle_sum = self.one_slice(x, lengths-2)
        vec_ang = self.conv_modules[1](angle_sum)

        abs_sum = self.one_slice(x, lengths-3)
        vec_abs = self.conv_modules[2](abs_sum)

        vec_gru = vec_gru.squeeze()
        vec_d = vec_d.squeeze()
        vec_ang = vec_ang.squeeze()
        vec_abs = vec_abs.squeeze()

        # print(vec_gru.shape, vec_d.shape, vec_ang.shape, vec_abs.shape)
        x = torch.cat([vec_gru, vec_d, vec_ang, vec_abs], dim=1)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        return self.lin(x) #torch.mean(x, dim=1, keepdim=True) # keepdim = True => {batchsize, 1}, otherwise => {batchsize}

class GRU_Conv2D_plus_model(GRU_Conv2D_model):
    def __init__(
            self,
            act_fn,
            fc_deep,
            conv2d_deep,
            dropout,
            conv_mid_ch_num,
            conv_out_ch_num,
            latent_tofu_ch):
        super().__init__(act_fn,
            fc_deep,
            conv2d_deep,
            dropout,
            conv_mid_ch_num,
            conv_out_ch_num,
            latent_tofu_ch)
        self.conv_modules = nn.ModuleList([
            self._create_conv_module(),
            self._create_conv_module(),
            self._create_conv_module()
        ])
        self.lin_last = self.create_lin_last(self.act_fn)
        
    def create_lin_last(self, act_fn):
        layers = []
        layers.append(nn.Linear(4, 32))
        layers.append(act_fn())
        for _ in range(self.fc_deep):
            layers.append(nn.Dropout(p=self.dropout))
            layers.append(nn.Linear(32, 32))
            layers.append(act_fn())
        layers.append(nn.Linear(32, 16))
        layers.append(act_fn())
        layers.append(nn.Linear(16, 1))
        return nn.Sequential(*layers)

    def _create_conv_module(self):
        conv3d = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=self.mid_channels, kernel_size=4, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=1, kernel_size=3, stride=1, padding=0, dilation=1),
                )
        return conv3d

    def one_slice(self, x, lengths):
        indices = lengths.view(-1, 1)
        expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8, 8)
        selected = torch.gather(x, 1, expanded_indices.to(x.device))
        return selected
    
    def forward(self, x, lengths):
        dhkl = self.one_slice(x, lengths-1)
        vec_d = self.conv_modules[0](dhkl)

        angle_sum = self.one_slice(x, lengths-2)
        vec_ang = self.conv_modules[1](angle_sum)

        abs_sum = self.one_slice(x, lengths-3)
        vec_abs = self.conv_modules[2](abs_sum)

        vec_d = vec_d.squeeze().unsqueeze(-1)
        vec_ang = vec_ang.squeeze().unsqueeze(-1)
        vec_abs = vec_abs.squeeze().unsqueeze(-1)

        x = super().forward(x, lengths-3)
        x_helper = torch.cat([x, vec_d, vec_ang, vec_abs], dim=1)

        return self.lin_last(x_helper) + x

class GRU_Conv2D_plusplus_model(GRU_Conv2D_model):
    def __init__(
            self,
            act_fn,
            fc_deep,
            conv2d_deep,
            dropout,
            conv_mid_ch_num,
            conv_out_ch_num,
            latent_tofu_ch):
        super().__init__(act_fn,
            fc_deep,
            conv2d_deep,
            dropout,
            conv_mid_ch_num,
            conv_out_ch_num,
            latent_tofu_ch)
        self.conv_modules = nn.ModuleList([
            self._create_conv_module(),
            self._create_conv_module(),
            self._create_conv_module()
        ])
        
        self.lin = self.create_lin_seq(self.act_fn, 12 + 1 * 8 * 1 * 1 * 3 * self.out_channels + (3 + 3) * 1 * self.out_channels)

    def _create_conv_module(self):
        conv3d = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=self.mid_channels, kernel_size=4, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=self.mid_channels, kernel_size=3, stride=1, padding=0, dilation=1),
                nn.BatchNorm3d(self.mid_channels),
                self.act_fn(),
                nn.Dropout3d(p=self.dropout),
                nn.Conv3d(in_channels=self.mid_channels, out_channels=4, kernel_size=3, stride=1, padding=0, dilation=1),
                )
        return conv3d

    def one_slice(self, x, lengths):
        indices = lengths.view(-1, 1)
        expanded_indices = indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8, 8)
        selected = torch.gather(x, 1, expanded_indices.to(x.device))
        return selected
    
    def forward(self, x, lengths):
        dhkl = self.one_slice(x, lengths-1)
        vec_d = self.conv_modules[0](dhkl)

        angle_sum = self.one_slice(x, lengths-2)
        vec_ang = self.conv_modules[1](angle_sum)

        abs_sum = self.one_slice(x, lengths-3)
        vec_abs = self.conv_modules[2](abs_sum)

        vec_d = vec_d.squeeze()
        vec_ang = vec_ang.squeeze()
        vec_abs = vec_abs.squeeze()

        x = super().forward_1st_step(x, lengths-3)
        x_helper = torch.cat([x, vec_d, vec_ang, vec_abs], dim=1)

        return self.lin(x_helper)
