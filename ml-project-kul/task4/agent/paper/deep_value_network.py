from tensorflow.keras import layers, models

class DeepValueNetwork(models.Model):
    def __init__(self, input_shape=(15, 15, 20), **kwargs):
        super(DeepValueNetwork, self).__init__(**kwargs)
        self.conv_layers = [
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='elu', input_shape=input_shape)
            for _ in range(7)  # Consider if more layers are needed based on performance
        ]
        self.dense_layers = [
            layers.Dense(256, activation='elu')  # Increased number of neurons
            for _ in range(2)  # You might consider adding more layers if necessary
        ]
        self.output_layer = layers.Dense(1, activation='sigmoid')  # Output for game situation evaluation

    def call(self, inputs, training=False):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = layers.Flatten()(x)
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return self.output_layer(x)

    def get_config(self):
        config = super(DeepValueNetwork, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
