#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 22:56:19 2025

@author: elmobarratt
"""

import numpy as np
import random

class GridWorld2D:
    def __init__(self, size=10, max_steps=40, obstacle_ratio=0.2):
        self.size = size
        self.max_steps = max_steps
        self.obstacle_ratio = obstacle_ratio
        self.reset()

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # Place obstacles
        num_obstacles = int(self.size * self.size * self.obstacle_ratio)
        obstacle_coords = random.sample([(i, j) for i in range(self.size) for j in range(self.size)], num_obstacles)

        for (i, j) in obstacle_coords:
            self.grid[i, j] = -1  # Obstacle

        # Choose start and goal positions not on obstacles
        empty_cells = [(i, j) for i in range(self.size) for j in range(self.size) if self.grid[i, j] == 0]
        self.start = random.choice(empty_cells)
        empty_cells.remove(self.start)
        self.goal = random.choice(empty_cells)

        self.agent_pos = self.start
        self.steps = 0

        return self._get_state()

    def _get_state(self):
        
        state = np.zeros((self.size,self.size,2))
        
        state[:,:,0] = self.grid                        # Inherit the grid encoding of -1 for blocked cells and 0 for free cells
        state[self.agent_pos[0],self.agent_pos[1],1]=1   # Encode position of the agent into state vector as 1
        
        goalDiff = np.array(self.goal)-np.array(self.agent_pos) # Get the vector bewteen the agent and the goal
        goalDiff = goalDiff/np.linalg.norm(goalDiff,ord=2)      # Normalise for stability

        return np.concatenate([goalDiff,state.reshape(-1)]) # Final dimension of state vector is 2 * self.size * self.size +2

    def _is_valid(self, i, j):
        return 0 <= i < self.size and 0 <= j < self.size and self.grid[i, j] != -1

    def step(self, action):
        self.steps += 1
        i, j = self.agent_pos

        if action == 0:     # up
            ni, nj = i - 1, j
        elif action == 1:   # down
            ni, nj = i + 1, j
        elif action == 2:   # left
            ni, nj = i, j - 1
        elif action == 3:   # right
            ni, nj = i, j + 1
        else:
            ni, nj = i, j

        if self._is_valid(ni, nj):
            self.agent_pos = (ni, nj)

        reward = 1 if self.agent_pos == self.goal else 0
        done = reward == 1 or self.steps >= self.max_steps

        return self._get_state(), reward, done

    def render(self):
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                pos = (i, j)
                if pos == self.agent_pos and pos == self.goal:
                    row += "AG "
                elif pos == self.agent_pos:
                    row += " A "
                elif pos == self.start:
                    row += " S "
                elif pos == self.goal:
                    row += " G "
                elif self.grid[i, j] == -1:
                    row += " # "
                else:
                    row += " . "
            print(row)
        print()



# --- Simple Policy Network ---
class PolicyNetwork:
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, lr=0.005):
        # Initialize weights
        self.W1 = np.random.randn(hidden_dim_1, input_dim) * 0.1
        self.b1 = np.zeros((hidden_dim_1,))
        self.W2 = np.random.randn(hidden_dim_2, hidden_dim_1) * 0.1
        self.b2 = np.zeros((hidden_dim_2,))
        self.W3 = np.random.randn(output_dim, hidden_dim_2) * 0.1
        self.b3 = np.zeros((output_dim,))
        self.lr = lr

    def forward(self, x):
        z1 = np.dot(self.W1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = np.tanh(z2)
        z3 = np.dot(self.W3, a2) + self.b3
        exp_logits = np.exp(z3 - np.max(z3))  # for stability
        probs = exp_logits / np.sum(exp_logits)
        return a1,a2, probs

    def update(self, trajectory):
        G = 0
        returns = []
        for _, _, r in reversed(trajectory):
            G = r + 0.99 * G
            returns.insert(0, G)
    
        for (s, a, _), G in zip(trajectory, returns):
            # Forward pass
            a1, a2, probs = self.forward(s)
    
            # Gradient of log-probability wrt z3 (output pre-softmax)
            dlog = -probs
            dlog[a] += 1  # policy gradient
            dlog *= G     # weight by return
    
            # Gradients for output layer
            dW3 = np.outer(dlog, a2)
            db3 = dlog
    
            # Backprop through a2 = tanh(z2)
            dz2 = np.dot(self.W3.T, dlog) * (1 - a2 ** 2)
            dW2 = np.outer(dz2, a1)
            db2 = dz2
    
            # Backprop through a1 = tanh(z1)
            dz1 = np.dot(self.W2.T, dz2) * (1 - a1 ** 2)
            dW1 = np.outer(dz1, s)
            db1 = dz1
    
            # Gradient ascent (policy gradient)
            self.W3 += self.lr * dW3
            self.b3 += self.lr * db3
            self.W2 += self.lr * dW2
            self.b2 += self.lr * db2
            self.W1 += self.lr * dW1
            self.b1 += self.lr * db1

# --- Training ---
env = GridWorld2D()
policy = PolicyNetwork(input_dim=2*10*10 + 2, hidden_dim_1=100,hidden_dim_2=50, output_dim=4)
#%%

score = []

for episode in range(2000):
    state = env.reset()
    trajectory = []
    total_reward = 0

    for t in range(env.max_steps):
        _,_, probs = policy.forward(state)
        action = np.random.choice(4, p=probs)
        # action = int(np.argmax(probs))
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        total_reward += reward
        if done:
            break

    policy.update(trajectory)
    score.append(total_reward)

    
    if episode % 20 == 0:
        score_percentage = round(sum(score)/len(score) *100 ,2)
        # print(f"Episode {episode}: Total Reward = {total_reward}")
        print(f"Episode {episode}: Success Rate = {score_percentage}%")
        
#%%

import pygame
import sys
import time

def run_pygame_env(env, policy):
    pygame.init()

    # Config
    size = env.size
    cell_size = 40
    margin = 2
    screen_size = size * (cell_size + margin) + margin
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Auto Agent with Click-to-Reset")

    clock = pygame.time.Clock()
    env.reset()
    done = False
    wait_for_click = False
    last_step_time = time.time()

    def draw_grid():
        screen.fill((30, 30, 30))
        for y in range(size):
            for x in range(size):
                rect = pygame.Rect(
                    x * (cell_size + margin) + margin,
                    y * (cell_size + margin) + margin,
                    cell_size,
                    cell_size
                )
                val = env.grid[y, x]
                color = (255, 255, 255)  # empty
                if (y, x) == env.agent_pos:
                    color = (0, 255, 0)  # agent = green
                elif (y, x) == env.goal:
                    color = (0, 0, 255)  # goal = blue
                elif val == -1:
                    color = (100, 100, 100)  # obstacle = gray
                pygame.draw.rect(screen, color, rect)

    while True:
        draw_grid()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if wait_for_click and event.type == pygame.MOUSEBUTTONDOWN:
                env.reset()
                done = False
                wait_for_click = False
                last_step_time = time.time()

        # Auto-step only if not waiting for user
        if not done and not wait_for_click and time.time() - last_step_time > 0.2:
            state = env._get_state()
            _, _, probs = policy.forward(state)
            action = np.random.choice(len(probs), p=probs)
            _,_, done = env.step(action)
            last_step_time = time.time()

            if done:
                wait_for_click = True

run_pygame_env(env,policy)
#%%
# print ("------------------------------")
# state = env.reset()


# for t in range(40):
    
#     env.render()
#     _,_, probs = policy.forward(state)
#     action = np.random.choice(4, p=probs)
#     next_state, reward, done = env.step(action)
    
#     if done:
#         break

# print("Success") if reward ==1 else print ("Failure")
