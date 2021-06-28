# Model Outputs

Store the model training outputs in this directory.
The following folder structure can be used to store them

```
outputs
|
|------ <some meaningful prefix>-<timestamp>
        |
        |
        |-------------keras_tensorboard
        |-------------output_model
|
|
|
.............................
...............
.............................
.......
```

- The meaningful prefix can be used to identify what model architecture and training hyperparamters were used in the model
- The timestamp is used to know the time of creation and to prevent two folders having same name
- keras_tensorboard folder is used for tensorboard callback and is used for visualization
- output_model folder is to store the model in order to use it for later
