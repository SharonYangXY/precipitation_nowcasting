from experiments.config import cfg
from collections import OrderedDict
from nowcasting.models.trajGRU import TrajGRU
from nowcasting.models.convLSTM import ConvLSTM

batch_size = cfg.GLOBAL.BATCH_SZIE
IN_LEN = cfg.RAIN.BENCHMARK.IN_LEN
OUT_LEN = cfg.RAIN.BENCHMARK.OUT_LEN

# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 120, 100), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 60, 50), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 30, 25), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 30, 25), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]

conv2d_params = OrderedDict({
    'conv1_relu_1': [6, 64, 7, 5, 1], # 100
    'conv2_relu_1': [64, 192, 5, 2, 2], # 33
    'conv3_relu_1': [192, 192, 3, 2, 1], #17
    'deconv1_relu_1': [192, 192, 4, 2, 1],  #N=(w-1)??s+k-2p 16*2+4-2*1=34
    'deconv2_relu_1': [192, 64, 4, 2, 1],   #33*3+5-2*1=100  99+5-2
    'deconv3_relu_1': [64, 64, 7, 5, 1], #99*5+5-2*0 = 500
    'conv3_relu_2': [64, 32, 3, 1, 1], # 499+1
    'conv3_3': [32, 6, 1, 1, 0]
})


# build model
convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 120, 100),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 60, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 30, 25),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}), # mark 1->2
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'final_conv3_3': [8, 1, 1, 1, 0]
        }),
    ],
    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 30, 25),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 60, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 120, 100),
                 kernel_size=3, stride=1, padding=1),
    ]
]

# build model
convlstm_multiChannel_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [3, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 120, 100),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 60, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 30, 25),
                 kernel_size=3, stride=1, padding=1),
    ]
]

convlstm_multiChannel_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 4, 2, 1]}), # mark 1->2
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'final_conv3_3': [8, 1, 1, 1, 0]
        }),
    ],
    [
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 30, 25),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 60, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 120, 100),
                 kernel_size=3, stride=1, padding=1),
    ]
]



# build model
fusion_convlstm_encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=64, b_h_w=(batch_size, 120, 100),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 60, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=192, num_filter=192, b_h_w=(batch_size, 30, 25),
                 kernel_size=3, stride=1, padding=1),
    ]
]


fusion_lstm_params = [
    [
        OrderedDict({'conv1_leaky_1': [2, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [32, 64, 5, 2, 2]}),
        OrderedDict({'conv3_leaky_1': [64, 64, 3, 2, 1]}),
    ],

    [
        ConvLSTM(input_channel=8, num_filter=32, b_h_w=(batch_size, 120, 100),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 60, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=64, num_filter=64, b_h_w=(batch_size, 30, 25),
                 kernel_size=3, stride=1, padding=1),
    ]
]

fusion_convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [256, 256, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [256, 96, 4, 2, 1]}), # mark 1->2
        OrderedDict({
            'deconv3_leaky_1': [96, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'final_conv3_3': [8, 1, 1, 1, 0]
        }),
    ],
    [
        ConvLSTM(input_channel=256, num_filter=256, b_h_w=(batch_size, 30, 25),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=256, num_filter=256, b_h_w=(batch_size, 60, 50),
                 kernel_size=3, stride=1, padding=1),
        ConvLSTM(input_channel=96, num_filter=96, b_h_w=(batch_size, 120, 100),
                 kernel_size=3, stride=1, padding=1),
    ]
]