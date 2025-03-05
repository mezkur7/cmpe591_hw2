# cmpe591_hw2

After Deadline extension

Important files:
hw2.py  
smoothing.ipynb  
policy_net_10000.pth  
target_net_10000.pth  
rps_per_episode.npy  
rewards_per_episode.npy  

10000 episodes with n_splits=30 

![Figure_1_10000](https://github.com/user-attachments/assets/e50384f2-2e9b-4b64-b436-fdefa8baf1ad)
![Figure_2_10000](https://github.com/user-attachments/assets/6dc4c175-3b74-47cd-bb98-689c1dbb3813)

![image](https://github.com/user-attachments/assets/74cb3747-616a-4626-9e18-352666376dfe)
![image](https://github.com/user-attachments/assets/532cce9d-ea4a-4ec6-af99-be0e75f35d26)

Limited Test result

My training ended this morning.
A little run I had:

Episode 1: Reward=28.44, RPS=0.5687  
Episode 2: Reward=37.38, RPS=0.7476  
Episode 3: Reward=38.93, RPS=0.7786  
Episode 4: Reward=8.20, RPS=0.1640  
Episode 5: Reward=2.92, RPS=0.0584  
![Figure_1_test](https://github.com/user-attachments/assets/aef37c6b-11b9-4500-9d9a-37ca6a7b8ee2)
![Figure_2_test](https://github.com/user-attachments/assets/1fd2052a-17d4-4f48-8258-21636c0bd0ee)

I believe some action results are still not that well. But the overall performance is very good.

Before Extension

Since my computer is very slow at calculating the episodes I had to lower the number of splits/steps in an episode. The performance could be better but I wanted to share the results of my experimentation.

2500 episodes with n_splits=10(parameter in step function in homework2.py file)

![Figure_1_2500](https://github.com/user-attachments/assets/474d3de3-b6d5-4e8c-bcfb-253f515e1376)
![Figure_2_2500](https://github.com/user-attachments/assets/ef99fdfa-441e-410f-a9ad-9399fa0117db)

![image](https://github.com/user-attachments/assets/15f4851b-f04e-4be9-be2e-a682b31b4823)
![image](https://github.com/user-attachments/assets/65887a16-26c1-4759-af9a-13bf262432cf)



5000 episodes with n_splits=10

![Figure_1_5000](https://github.com/user-attachments/assets/c296ffd8-907d-4a40-8025-1eb74898ca84)
![Figure_2_5000](https://github.com/user-attachments/assets/26f8fe58-91d0-4033-95dd-a55ea8b31a08)

![image](https://github.com/user-attachments/assets/b753334a-394f-4444-b0ca-c42beddd4be9)
![image](https://github.com/user-attachments/assets/dbc8d087-5de1-4c30-b52c-7dd281b44dc1)



2500-5000 episodes with n_splits=30 with hw2_retrain.py(step function in homework2.py file)

![Figure_1_2500-5000](https://github.com/user-attachments/assets/388fe4cb-8864-4be3-8d83-da25d7d5b933)
![Figure_2_2500-5000](https://github.com/user-attachments/assets/7361b11f-5ab7-439d-aefd-98ff42246790)

![image](https://github.com/user-attachments/assets/847a0d37-9237-428b-bcaf-d143613fdd8a)
![image](https://github.com/user-attachments/assets/12bd9f26-2ba4-415b-9146-3a63ba139ef5)

After the deadline extension

10000 


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
