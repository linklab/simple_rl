# Installation for Linux

- Create environment
  ```commandline
  conda create -n simple_rl python==3.9
  ```

- Activate environment
  ```commandline
  conda activate simple_rl
  ```

- Install pytorch
  - go to https://pytorch.org/ and install pytorch
  - check cuda if you installed cuda-support version
    ```commandline
    python -c "import torch; print(torch.cuda.is_available()); print(torch.rand(2,3).cuda());"
    ```

- Install package
  ```commandline
  pip install gym==0.22.0
  pip install pygame==2.1.2
  conda install -c conda-forge swig==4.0.2
  conda install -c conda-forge box2d-py==2.3.8
  pip install ale-py==0.7.4
  pip install gym[accept-rom-license]==0.4.2
  pip install opencv-python==4.5.5.62
  pip install lz4==4.0.0
  conda install -c fastai nvidia-ml-py3==7.352.0
  conda install -c plotly plotly==5.6.0
  conda install -c conda-forge wandb
  ```

- run files
  - ```a_q_learning/env_info.py```
  - ```a_q_learning/table_q_learning.py```
  - .
  - ```b_dqn/cartpole_dqn/env_info.py```
  - ```b_dqn/cartpole_dqn/dqn_train.py```
  - ```b_dqn/cartpole_dqn/dqn_play.py```
  - .
  - ```b_dqn/pong_dqn/gym_info.py```
  - ```b_dqn/pong_dqn/gym_atari_game_img.py```
  - ```b_dqn/pong_dqn/gym_info.py```
  - ```b_dqn/pong_dqn/dqn_cnn_train.py```
  - ```b_dqn/pong_dqn/dqn_cnn_play.py```
  - .
  - ```b_dqn/task_scheduling_dqn/dqn_train.py```