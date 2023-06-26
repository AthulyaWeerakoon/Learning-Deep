import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

input_count = 4
action_count = 2
hidden_neuron_count = 128


def create_model(num_inputs, num_actions, num_hidden):
    inputs = layers.Input(shape=(num_inputs,))
    # hidden = layers.Dense(num_hidden, activation='relu')(inputs)
    common = layers.Dense(num_hidden, activation='relu')(inputs)
    _action = layers.Dense(num_actions, activation='softmax')(common)
    critic = layers.Dense(1)(common)

    return keras.Model(inputs=inputs, outputs=[_action, critic])


# setting up virtual environment
gamma = 0.99
max_steps_per_episode = 10000
env = gym.make("CartPole-v1", render_mode="human")
eps = np.finfo(np.float32).eps.item()

model = create_model(input_count, action_count, hidden_neuron_count)

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss = keras.losses.Huber()
action_hist = []
critic_hist = []
reward_hist = []
running_reward = 0
episode_count = 0

while True:
    state, _ = env.reset(seed=42)
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            env.render()

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs, critic_val = model(state)
            critic_hist.append(critic_val[0, 0])

            action: int = np.random.choice(action_count, 1, p=np.squeeze(action_probs))[0]
            action_hist.append(tf.math.log(action_probs[0, action]))

            state, reward, terminated, truncated, info = env.step(action)
            reward_hist.append(reward)
            episode_reward += reward

            if terminated:
                break

        running_reward = 0.05 * episode_reward + 0.95 * running_reward

        returns = []
        discounted_sum = 0
        for r in reward_hist[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        hist = zip(action_probs, critic_val, returns)
        actor_losses = []
        critic_loses = []

        for log_prob, val, ret in hist:
            diff = ret - val
            actor_losses.append(-log_prob * diff)

            critic_loses.append(
                loss(tf.expand_dims(val, 0), tf.expand_dims(ret, 0))
            )

        loss_val = sum(actor_losses) + sum(critic_loses)
        grads = tape.gradient(loss_val, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        action_hist.clear()
        critic_hist.clear()
        reward_hist.clear()

    episode_count += 1
    if episode_count % 10 == 0:
        print("running reward: {:.2f} at episode {}".format(running_reward, episode_count))

    if running_reward > 195:
        print("Solved at episode {}!".format(episode_count))
        break
