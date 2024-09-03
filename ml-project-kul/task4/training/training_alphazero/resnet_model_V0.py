import os
import logging
log = logging.getLogger(__name__)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Add, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.conv2d_1 = Conv2D(num_filters, kernel_size, padding='same')
        self.bn_1 = BatchNormalization()
        self.conv2d_2 = Conv2D(num_filters, kernel_size, padding='same')
        self.bn_2 = BatchNormalization()

    def call(self, inputs, training=False):
        x = self.conv2d_1(inputs)
        x = self.bn_1(x, training=training)
        x = Activation('relu')(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x, training=training)
        x = Add()([x, inputs])
        return Activation('relu')(x)

class AlphaZeroModel(Model):
    def __init__(self, input_shape, output_size, nn_width, nn_depth, weight_decay, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.input_layer = tf.keras.Input(shape=input_shape)
        self.conv = Conv2D(nn_width, 3, padding='same')
        self.bn = BatchNormalization()
        self.res_blocks = [ResidualBlock(nn_width, 3) for _ in range(nn_depth)]
        self.policy_head = self._build_policy_head(output_size)
        self.value_head = self._build_value_head(nn_width)
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss={'policy': CategoricalCrossentropy(), 'value': MeanSquaredError()},
                     loss_weights={'policy': 0.5, 'value': 0.5})

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = Activation('relu')(x)
        for res_block in self.res_blocks:
            x = res_block(x, training=training)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return {'policy': policy, 'value': value}

    def _build_policy_head(self, output_size):
        policy_conv = Conv2D(2, 1, name="policy_conv")
        policy_bn = BatchNormalization(name="policy_batch_norm")
        policy_activation = Activation('relu')
        policy_flatten = Flatten()
        policy_dense = Dense(output_size, activation='softmax', name="policy_output")
        return tf.keras.Sequential([policy_conv, policy_bn, policy_activation, policy_flatten, policy_dense])

    def _build_value_head(self, nn_width):
        value_conv = Conv2D(1, 1, name="value_conv")
        value_bn = BatchNormalization(name="value_batch_norm")
        value_activation = Activation('relu')
        value_flatten = Flatten()
        value_dense = Dense(nn_width, activation='relu')
        value_output = Dense(1, activation='tanh', name="value_output")
        return tf.keras.Sequential([value_conv, value_bn, value_activation, value_flatten, value_dense, value_output])



class NNetWrapper:
    def __init__(self, game, path, learning_rate, nn_width, nn_depth, weight_decay, config):
        self.game = game
        num_rows_input = game.get_parameters()['num_rows'] + 1
        num_cols_input = game.get_parameters()['num_cols'] + 1
        self.input_shape = (3, num_rows_input * num_cols_input, 3)
        self.action_size = game.num_distinct_actions()  
        self.strategy = get_distribution_strategy() 
     
        with self.strategy.scope():
            self.model = AlphaZeroModel(
                input_shape=self.input_shape, 
                output_size=self.action_size, 
                nn_width=nn_width, 
                nn_depth=nn_depth, 
                weight_decay=weight_decay
            )
            self.model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss={'policy': CategoricalCrossentropy(), 'value': MeanSquaredError()},
                loss_weights={'policy': 0.5, 'value': 0.5}
            )
        self.path = path
        self.config = config


    def train(self, examples, batch_size=16):
        with self.strategy.scope():
            log.info(f'Using strategy with devices: {self.strategy.num_replicas_in_sync}')
            inputs, target_policy, target_value = list(zip(*examples))
            input_boards = np.asarray([np.array(board, dtype=np.float32) for board in inputs])
            target_policy = np.asarray(target_policy)
            
            
            target_value = np.asarray(target_value)

            inputs = np.array([board.reshape(self.input_shape) if board.shape != self.input_shape else board for board in input_boards])

            if inputs.ndim != 4:
                raise ValueError(f"Input dimension mismatch. Expected 4 dimensions, got {inputs.ndim}")

            max_batch_size = 128 
            required_batch_size = self.strategy.num_replicas_in_sync * batch_size
            
            if len(inputs) % required_batch_size != 0 or batch_size > max_batch_size:
                adjusted_batch_size = min(max_batch_size, (len(inputs) // self.strategy.num_replicas_in_sync) * self.strategy.num_replicas_in_sync)
                log.warning(f"Adjusting batch size from {batch_size} to {adjusted_batch_size} to fit GPU requirements.")
                batch_size = adjusted_batch_size
        
            log.info(f"Total input size: {len(inputs)}, Batch size: {batch_size}, Number of GPUs: {self.strategy.num_replicas_in_sync}")
            
            if len(inputs) < batch_size * self.strategy.num_replicas_in_sync:
                raise ValueError("Not enough data to distribute across GPUs properly.")

            if len(inputs) == 0:
                raise ValueError("Processed data resulted in an empty dataset.")
            
            if len(inputs) % batch_size != 0:
                log.info(f"Adjusting batch size from {batch_size} due to dataset size.")
                batch_size = len(inputs) // (len(inputs) // batch_size)  

            dataset = tf.data.Dataset.from_tensor_slices((inputs, {'policy': target_policy, 'value': target_value}))

            dataset = dataset.shuffle(buffer_size=len(inputs))
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat()
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            
            try:
                log.info("Started training the model.")           
                self.model.fit(
                    dataset, 
                    epochs=self.config['epochs'], 
                    steps_per_epoch=len(inputs) // batch_size,
                    verbose=False
                )

                losses = self.model.history.history['loss']
                log.info(f"Training losses: {losses}")
            except tf.errors.OutOfRangeError:
                pass

    def predict(self, state):
        input = np.array(state.observation_tensor(), dtype=np.float32)
        if input.ndim != 4 or input.shape[1:] != self.input_shape:
            input = input.reshape((1,) + self.input_shape)
        try:
            result = self.model.predict(input, verbose=False)
        except tf.errors.OutOfRangeError:
                pass
        return result['policy'][0], result['value'][0][0]

    def save_checkpoint(self, folder, filename):
        full_path = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Creating checkpoint directory at {folder}")
        self.model.save_weights(full_path)
        log.info(f"Checkpoint saved at {full_path}")

    def load_checkpoint(self, folder, filename):
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            self.model.load_weights(full_path)
            log.info(f"Loaded weights from {full_path}")
        else:
            log.info(f"No checkpoint found at {full_path}, please check the file path.")
    
_strategy_instance = None

def get_distribution_strategy():
    global _strategy_instance
    if _strategy_instance is None:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    log.error("Failed to set memory growth: {}".format(e))
                _strategy_instance = tf.distribute.MirroredStrategy()
            else:
                _strategy_instance = tf.distribute.get_strategy()
        except RuntimeError as e:
            log.error(e)
            _strategy_instance = tf.distribute.get_strategy()
    return _strategy_instance

