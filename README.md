# cmpe591_hw2

Since my computer is very slow at calculating the episodes I had to lower the number of splits/steps in an episode. The performance could be better but I wanted to share the results of my experimentation.

2500 episodes with n_splits=10(parameter in step function in homework2.py file)

![Figure_1_2500](https://github.com/user-attachments/assets/474d3de3-b6d5-4e8c-bcfb-253f515e1376)
![Figure_2_2500](https://github.com/user-attachments/assets/ef99fdfa-441e-410f-a9ad-9399fa0117db)

5000 episodes with n_splits=10

![Figure_1_5000](https://github.com/user-attachments/assets/c296ffd8-907d-4a40-8025-1eb74898ca84)
![Figure_2_5000](https://github.com/user-attachments/assets/26f8fe58-91d0-4033-95dd-a55ea8b31a08)

2500-5000 episodes with n_splits=30(step function in homework2.py file)

![Figure_1_2500-5000](https://github.com/user-attachments/assets/388fe4cb-8864-4be3-8d83-da25d7d5b933)
![Figure_2_2500-5000](https://github.com/user-attachments/assets/7361b11f-5ab7-439d-aefd-98ff42246790)

Hyperparameter Set
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_DECAY_ITER = 5
EPSILON_MIN = 0.05
LEARNING_RATE = 0.001
BATCH_SIZE = 64
UPDATE_FREQ = 10
TARGET_NETWORK_UPDATE_FREQ = 50 / 200 (retrain run)
BUFFER_LENGTH = 100000
N_ACTIONS = 8
