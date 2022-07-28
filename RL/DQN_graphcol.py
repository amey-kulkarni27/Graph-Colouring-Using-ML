import numpy as np
import keras.backend as backend
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import pickle
import os
from PIL import Image
import cv2
import networkx as nx

import sys
sys.path.insert(0, '/home/amey.kulkarni/Graph-Colouring-Using-ML/ML')
from generate_kpart import gen_kpart, display_graph
from feature_vector import feature_vector
from operations import operations, vertex_pair_non_edge, get_action

# Calculate loss per step
# Print steps per episode

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training (basically batch size)
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'N=25K=4'
MIN_REWARD = -80  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 2_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

class Graph:
    def __init__(self, k, n, density):
        self.G, self.coords = gen_kpart(k, n, density)
        # display_graph(G, coords)
    def update_fv(self, method, fv_len):
        self.fv = feature_vector(self.G, method=method, k=fv_len)
    def dist(self, u, v):
        # print(len(self.G.nodes), len(self.fv))
        return np.linalg.norm(self.fv[u] - self.fv[v])

class GraphEnv:
    K = 4 # colours
    N = 25 # vertices of each colour
    DENSITY = 0.3
    FV_LEN = 4 # length of feature vector
    METHOD = 'topk'
    NUM_ACTIONS = 3 # 3rd action is don't do anything
    UPDATE_INTERVAL = 1 # update the feature vector after every _ turns
    p = 5 # We select the best pair out of the top p

    REWARD = 50
    PENALTY = 100
    TURN = 1
    THRESH = 1
    OBSERVATION_SPACE_VALUES = FV_LEN  # 4
    ACTION_SPACE_SIZE = NUM_ACTIONS
    MAX_STEPS = 100

    def best_of_p(self):
        best_pair = None
        min_dist = 0
        for _ in range(self.p):
            nxt_nodes = vertex_pair_non_edge(self.G_obj.G)
            if best_pair == None:
                best_pair = nxt_nodes
                min_dist = np.linalg.norm(self.G_obj.fv[nxt_nodes[1]] - self.G_obj.fv[nxt_nodes[0]])
            else:
                dist = np.linalg.norm(self.G_obj.fv[nxt_nodes[1]] - self.G_obj.fv[nxt_nodes[0]])
                if dist < min_dist:
                    best_pair = nxt_nodes
                    min_dist = dist
        return nxt_nodes

    def reset(self):
        self.G_obj = Graph(self.K, self.N, self.DENSITY)

        self.episode_step = 0
        self.cnt = 0
        self.G_obj.update_fv(self.METHOD, self.FV_LEN) # create fv
        # Select nodes with the minimum distance among p pairs
        nodes = self.best_of_p()
        observation = abs(self.G_obj.fv[nodes[1]] - self.G_obj.fv[nodes[0]]).reshape(1, -1)
        observation = tuple(observation.reshape(-1))
        return observation

    def step(self, action):
        self.episode_step += 1
        cols = self.N * self.K # returning number of colours
        if action < 2:
            node_pair = vertex_pair_non_edge(self.G_obj.G)
            self.G_obj.G = operations(self.G_obj.G, action, node_pair)
            N = len(self.G_obj.G.nodes)
            mapping = {old: new for (old, new) in zip(self.G_obj.G.nodes, [i for i in range(N)])}
            self.G_obj.G = nx.relabel_nodes(self.G_obj.G, mapping)
            # print(len(self.G_obj.G.nodes), action)
            self.cnt += 1
            self.cnt %= self.UPDATE_INTERVAL
            if self.cnt == 0:
                self.G_obj.update_fv(self.METHOD, self.FV_LEN)

        if (vertex_pair_non_edge(self.G_obj.G)) == False:
            # If a clique has been formed
            edges = random.sample(self.G_obj.G.edges(), 1)
            nxt_nodes = (min(edges[0]), max(edges[0]))
        else:
            # Select nodes with the minimum distance among p pairs
            nxt_nodes = self.best_of_p()
        new_observation = abs(self.G_obj.fv[nxt_nodes[1]] - self.G_obj.fv[nxt_nodes[0]]).reshape(1, -1)


        if (vertex_pair_non_edge(self.G_obj.G)) == False:
            cols = len(self.G_obj.G.nodes())
            d_correct = cols - self.K
            reward = self.REWARD // pow(2, d_correct)
            print(cols, reward)
            if reward < self.THRESH:
                reward = -self.PENALTY
        else:
            reward = -self.TURN
        done = False
        if (vertex_pair_non_edge(self.G_obj.G)) == False or self.episode_step >= self.MAX_STEPS:
            done = True

        new_observation = tuple(new_observation.reshape(-1))
        return new_observation, reward, done, self.episode_step, cols

    def render(self):
        display_graph(self.G_obj.G, self.G_obj.coords)


env = GraphEnv()

# For stats
ep_rewards = [-200]
steps_list = [env.MAX_STEPS]
cols_list = [env.N * env.K]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()


# Agent class
class DQNAgent:
    def __init__(self, load):
        if load:
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into new model
            self.model.load_weights("model.h5")
            self.model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
            print("Loaded model from disk")
        else:
            # Main model
            self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        # Hidden layer 1
        model.add(Dense(64, input_dim = env.OBSERVATION_SPACE_VALUES , activation = 'relu'))
        # Hidden layer 2
        model.add(Dense(64, activation = 'relu'))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (3)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        state = (np.asarray(state)).reshape(-1, env.OBSERVATION_SPACE_VALUES)
        return self.model.predict(state)

load = False
agent = DQNAgent(load)
streak = 0
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from DQN
            action = np.argmax(agent.get_qs(current_state))
            if action == 2:
                streak += 1
                if streak == 2:
                    action = 1 ^ np.argmin(agent.get_qs(current_state))
                    streak = 0
            else:
                streak = 0
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done, tot_steps, final_cols = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    steps_list.append(tot_steps)
    cols_list.append(final_cols)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        avg_steps = sum(steps_list[-AGGREGATE_STATS_EVERY:])/len(steps_list[-AGGREGATE_STATS_EVERY:])
        avg_cols = sum(cols_list[-AGGREGATE_STATS_EVERY:])/len(cols_list[-AGGREGATE_STATS_EVERY:])
        print(tot_steps)
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, steps=avg_steps, cols=avg_cols)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    if episode % AGGREGATE_STATS_EVERY == 0:
        model_json = agent.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        agent.model.save_weights("model.h5")
        print("Saved model to disk")

# Saving model
model_json = agent.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
agent.model.save_weights("model.h5")
print("Saved model to disk")

# 1) Let the observation space be a representation (something like Graph2vec) of the graph
# 2) Feature Vector for the graph (i) Graph2Vec, (ii) Hand-crafted feature vector
# 3) Graph2Vec -> (i) can't be permutationally invariant. (ii) action space keeps changing
# 4) Sequence model / Pointer Networks