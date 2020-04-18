from torch import nn
import torch


class FusionEncoder(nn.Module):
    def __init__(self, image_enconder, typh_encoder):
        super().__init__()
        self.image_enconder = image_enconder
        self.typh_encoder = typh_encoder

    # input: 5D S*B*I*H*W
    def forward(self, input):
        output_precipitation = self.image_enconder(input[:, :, 0:1, :, :])
        output_typh = self.typh_encoder(input[:, :, 1:3, :, :])
        output = []
        for i in range(len(output_precipitation)):
            precipitation_h = output_precipitation[i][0]
            precipitation_c = output_precipitation[i][1]
            typh_h = output_typh[i][0]
            typh_c = output_typh[i][1]
            output_h = torch.cat((precipitation_h, typh_h), 1)
            output_c = torch.cat((precipitation_c, typh_c), 1)
            output.append((output_h, output_c))
        return tuple(output)
