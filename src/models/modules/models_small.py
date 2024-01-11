import math

from .Improved_conformer import ConformerBlock
from .attention import *
from .conv_modules import *


class MaskGate(nn.Module):

    def __init__(self, channels):
        super().__init__()
        
        self.output      = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Tanh())
        self.output_gate = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.Sigmoid())
        self.mask        = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), nn.ReLU())

    def forward(self, x):

        mask = self.output(x) * self.output_gate(x)
        mask = self.mask(mask)
    
        return mask
    

    
class Encoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head, layer, depth):
        super().__init__()
        
        self.layer      = layer
        self.depth      = depth
        self.down_conv  = nn.Sequential(nn.Conv1d(in_channels, in_channels, kernel_size, stride), nn.BatchNorm1d(in_channels), nn.ReLU())
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=2)

    def forward(self, x):

        x = self.down_conv(x)        
        x = self.conv_block(x)

        return x
    
class Decoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, segment_len, head, layer, depth):
        super().__init__()
        
        self.layer      = layer
        self.depth      = depth
        self.conv_block = ResConBlock(in_channels, growth1=2, growth2=1/2)
        self.up_conv    = nn.Sequential(nn.ConvTranspose1d(out_channels, out_channels, kernel_size, stride),
                                     nn.BatchNorm1d(out_channels), nn.ReLU())


    def forward(self, x):
        
        x = self.conv_block(x)
        x = self.up_conv(x)

        
        return x
    
class T_Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden, depth, kernel_size, stride, growth, head, segment_len):
        super().__init__()
        
        self.depth    = depth      
        self.in_conv  = nn.Sequential(nn.Conv1d(in_channels, hidden, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm1d(hidden), nn.ReLU())
        self.out_conv = nn.Sequential(nn.Conv1d(hidden, in_channels, kernel_size=3, stride=1, padding=1))      
        in_channels   = in_channels*hidden
        out_channels  = out_channels*growth
        
        encoder = []
        decoder = []
        for layer in range(depth):
            encoder.append(Encoder(in_channels, out_channels*hidden, kernel_size, stride, segment_len, head, layer, depth))
            decoder.append(Decoder(out_channels*hidden, in_channels, kernel_size, stride, segment_len, head, layer, depth))

            in_channels  = hidden*(2**(layer+1))
            out_channels *= growth
            
        decoder.reverse()
        
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)      
        
        hdim           = (hidden*growth**(layer+1))
        self.linear    = nn.Sequential(nn.Linear(hdim, hdim, bias = False), nn.ReLU())
        self.mask_gate = MaskGate(hidden)
            
    def forward(self, x):
        
        x       = self.in_conv(x)
        enc_out = x
        
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)
            
        x = x.permute(0, 2, 1) # (B,N,T) -> (B,T, N)
        x = self.linear(x)
        x = x.permute(0, 2, 1) # (B,N,T)

        for decoder in self.decoder:
            skip = skips.pop(-1)
            x    = x + skip[..., :x.shape[-1]]
            x    = decoder(x)        

        mask = self.mask_gate(x)
        x    = enc_out * mask
        x    = self.out_conv(x)
        
        return x

def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


class T_Domain_Block(nn.Module):
    eps = 1e-3
    rescale = 0.1
    
    def __init__(self, in_channels, out_channels, hidden, depth, kernel_size, stride, growth, 
                       head, segment_len):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride      = stride
        self.depth       = depth
        
        self.T_Block = T_Block(in_channels, out_channels, hidden, depth, kernel_size, stride,
                               growth, head, segment_len)

        print('---rescale applied---')
        rescale_module(self, reference=self.rescale)

    def padding(self, length):

        length = math.ceil(length)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length))
        
        return int(length)

    def forward(self, x):
        
        x2  = x.mean(dim=1, keepdim=True)
        std = x2.std(dim=-1, keepdim=True)
        x   = x / (self.eps + std)
        
        # x (B, 1, T)
        length = x.shape[-1]
        x      = F.pad(x, (0, self.padding(length) - length))
        x      = self.T_Block(x)
        x      = x[..., :length]
        
        return std * x



class CRC(nn.Module):
    def __init__(self, num_channel=64):
        super(CRC, self).__init__()
        self.CRC_T = ConformerBlock(dim=num_channel, dim_head=num_channel // 4, heads=4,
                                    conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.CRC_F = ConformerBlock(dim=num_channel, dim_head=num_channel // 4, heads=4,
                                    conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_t = self.CRC_T(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.CRC_F(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f


class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class TF_Encoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(TF_Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)


class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x



class TF_Domain_Block(nn.Module):
    def __init__(self, num_channel, num_features):
        super(TF_Domain_Block, self).__init__()

        self.encoder = TF_Encoder(in_channel=3, channels=num_channel)

        self.CRC1 = CRC(num_channel=num_channel)
        self.CRC2 = CRC(num_channel=num_channel)
        self.CRC3 = CRC(num_channel=num_channel)
        self.CRC4 = CRC(num_channel=num_channel)

        self.decoder_mask = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.decoder_complex = ComplexDecoder(num_channel=num_channel)

    def forward(self, x_in):

        out_1 = self.encoder(x_in)

        out_2 = self.CRC1(out_1)
        out_3 = self.CRC2(out_2)
        out_4 = self.CRC3(out_3)
        out_5 = self.CRC4(out_4)

        mask = self.decoder_mask(out_5)
        complex = self.decoder_complex(out_5)

        return mask, complex