# Linear Probe Classification Results

This directory contains the results of the linear probe classification experiments. The results are stored as json files with the following format:

```
[
    [
        model_name,
        dataset_name,
        step,
        {
            layer_name: accuracy,
            ...
        }
    ],
    ...
]
```


# Notes for different files

## step0

```
step=0
batch_size=128
epochs=80
shuffle=True
opt=Adam(1e-3)
resize=None
```

## step50

```
step=50
batch_size=128
epochs=80
shuffle=True
opt=Adam(1e-3)
resize=None
```

## step50_resize128

```
step=50
batch_size=128
epochs=80
shuffle=True
opt=Adam(1e-3)
resize=128
```

## step50_resize256

```
step=50
batch_size=128
epochs=80
shuffle=True
opt=Adam(1e-3)
resize=256
```

## step50_resize512

```
step=50
batch_size=128
epochs=80
shuffle=True
opt=Adam(1e-3)
resize=512
```

## step100

```
step=100
batch_size=32
epochs=50
shuffle=True
opt=Adam(1e-3)
resize=None
```


## step500

```
step=500
batch_size=32
epochs=50
shuffle=True
opt=Adam(1e-3)
resize=None
```
