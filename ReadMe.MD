# Latency Test


## Installation

```
git clone https://github.com/kennethzhao24/latency_test
cd latency_test
pip intall -r requirements.txt
```


## Run Inference
```
bash inference.sh
```


## Apple Sillicon (M1)


|      | ZAD-60M | ZAD-80M | ZAD-100M | OPT-125M | OPT-350M | Pythia-70M | Pythia-160M | Cerebras-GPT | GPT-2  |
| :--: | :-----: | :-----: | :------: | :------: | :------: | :--------: | :---------: | :----------: | :----: |
|  64  |  13.3   |  20.9   |   34.0   |   52.5   |  143.5   |    24.1    |    63.6     |     46.9     |  67.3  |
| 128  |  23.0   |  37.9   |   58.0   |   74.3   |  222.5   |    36.0    |    89.0     |     76.6     | 114.8  |
| 256  |  54.0   |  78.6   |  115.6   |  133.7   |  366.5   |    55.9    |    150.5    |    135.8     | 235.2  |
| 512  |  132.4  |  202.9  |  302.6   |  270.7   |  793.8   |   117.9    |    325.8    |    278.2     | 551.7  |
| 1024 |  422.5  |  621.0  |  935.9   |  756.4   |  2783.6  |   317.2    |    822.8    |    682.9     | 1764.1 |


## NVIDIA Jetson Nano