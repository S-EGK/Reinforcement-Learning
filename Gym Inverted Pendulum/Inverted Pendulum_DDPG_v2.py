#%% Libraries
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Create the Environment
problem = "Pendulum-v1"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print(f"Size of the Sate Pace -> {num_states}")
num_actions = env.action_space.shape[0]
print(f"Size of Action Space -> {num_actions}")

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

#%% Ornstein-Uhlenbeck process for generating noise
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev =std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
            )
        # Store x into x_prev
        # Makes noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

#%% Buffer class for Experience Replay
class Buffer:
    def __init__(self, buffer_capacity=100_000, batch_size=64):
        # number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Number of tuples to train on
        self.batch_size = batch_size

        # This tells us num of times record() was called
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch,):
        # Training and updating the Actor & Critic networks
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # used '-value' as we want to maximize the values given by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    # We compute the loss and update parameters
    def learn(self):
        # get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

# this update target parameter slowly based on rate 'tau', whichb is much less than one
@tf.function
def update_target(target_weights, weights, tau):
    for (a,b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

#%% Define Actor and Critic Networks
def get_actor():
    # initialize weights between -3e-3 and 3e-3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

    # our upper bound is 2.0 for pendulum
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)

    return model

def get_critic():
    # state as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # both are passed through seperate layers before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # outputs single value for given state-action pair
    model = tf.keras.Model([state_input, action_input], outputs)

    return model

#%% Action Policy
def policy(state, noise_object):
    # print(state)
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

#%% Training hyperparameters
std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 150
# discount factor for future rewards
gamma = 0.99
# used to update target networks
tau = 0.005

buffer = Buffer(50_000, 64)

#%% Training Loop
# to store reward history of each episode
ep_reward_list = []
# to store average reward history of last few episodes
avg_reward_list = []

# takes about 20 min to train
for ep in tqdm(range(1, total_episodes+1), ascii=True, unit="episode"):
    prev_state = env.reset()
    episodic_reward = 0
    step = 0
    while True:
        step += 1
        # print(step)
        env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        # print(tf_prev_s1tate)
        action = policy(tf_prev_state, ou_noise)
        # recieve state and reward from environment
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # end this episode when 'done' is true
        if done or step > 200:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")
    avg_reward_list.append(avg_reward)

env.close()

# Plotting graph
# episodes vs Avg Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg Episodic Reward")
plt.show()

# Save the weights
actor_model.save_weights("pendulum ddpg/pendulum_actor.hr")
critic_model.save_weights("pendulum ddpg/pendulum_critic.h5")

target_actor.save_weights("pendulum ddpg/pendulum_target_actor.h5")
target_critic.save_weights("pendulum ddpg/pendulum_target_critic.h5")