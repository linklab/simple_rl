# Installation for Windows
Notice: Gym do not officially support Windows

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
  pip install gym[atari]==0.22.0
  pip install gym[accept-rom-license]==0.22.0
  conda install -c conda-forge box2d-py==2.3.8
  pip install pygame==2.1.0
  pip install opencv-python==4.5.5.62
  pip install lz4==4.0.0
  pip install plotly==5.6.0
  conda install -c conda-forge wandb
  ```