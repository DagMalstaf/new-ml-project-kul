from tensorflow.keras import layers, models, regularizers

class DeepStrategyNetwork(models.Model):
    def __init__(self, input_shape=(15, 15, 20), num_actions=120, **kwargs):
        super(DeepStrategyNetwork, self).__init__(**kwargs)
        self.conv_layers = [
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='elu', input_shape=input_shape)
            for _ in range(5)  # You might consider increasing the number of layers or the number of filters
        ]
        
        self.dense1 = layers.Dense(512, activation='elu')  # Increased number of neurons
        self.dropout = layers.Dropout(0.2)  # Maintain dropout to prevent overfitting
        self.output_layer = layers.Dense(num_actions, activation='softmax')  # Adjusted number of actions

    def call(self, inputs, training=False):
        x = inputs
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = layers.Flatten()(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        config = super(DeepStrategyNetwork, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
