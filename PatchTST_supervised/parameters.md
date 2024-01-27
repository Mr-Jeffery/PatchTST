# Dataloader

- `embed`
    1. `timeF`: This is a form of temporal feature encoding. In this method, the model adds temporal features to the time series values. These serve as “positional encodings” for the Transformer encoder and decoder. Examples are “day of the month”, “month of the year”, etc. as scalar values (and then stacked together as a vector). For instance, if a given time-series value was obtained on the 11th of August, then one could have [11, 8] as time feature vector (11 being “day of the month”, 8 being “month of the year”).

    2. `fixed`: This refers to fixed positional encoding, which is used in vanilla transformers. In this method, the position encoding is fixed and does not change during training. It's often used to inject information about the relative or absolute position of the tokens in the sequence.

    3. `learned`: This refers to learned positional encoding. Unlike fixed encoding, learned positional encoding allows transformers to be more flexible and better exploit sequential ordering information. The model learns the positional encodings during training, which can potentially lead to a better representation of the time information.

    Each of these methods has its own advantages and is used based on the specific requirements of the time series analysis task.

# PatchTST
- `decomposition`
    Decomposite the time series into two parts: trend and residual. This can help mitigate distribution shifting problem.

    Trend is computed by applying 1d convolution with kernel function whose size is determined by `kernel_size`.

- `kernel_size`
    Determine the length of the kernel, i.e. the length of time to be taken into consider when smoothing the time series.

- `revin`
    Whether to use [RevIN](https://github.com/ts-kim/RevIN).

# Formers 
- `embed_type`
    0: default 
    1: value embedding + temporal embedding + positional embedding 
    2: value embedding + temporal embedding 3: value embedding + positional embedding 
    4: value embedding'
- `patch_len`
    Similar to tokens in original transformer and image batch in ViT, a time series sample is broken into shorter parts called patches, these patches might overlap or not.
- `strides`
    Strides are the non overlaping part of the patches
- `activation`
    'relu' or 'gelu' as activation funciton

