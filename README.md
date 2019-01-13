# 2048 homework
# How to run
* 运行evaluate(50次): 
*     需先修改[`agents.py`](game2048/agents.py) 里面load 的网络(mymodel2.pth)路径
* python evaluate.py

# Code structure
* [`game2048/`](game2048/): the main package.
    * [`game.py`](game2048/game.py): the core 2048 `Game` class.
    * [`agents.py`](game2048/agents.py): the `Agent` class with instances.
    * [`displays.py`](game2048/displays.py): the `Display` class with instances, to show the `Game` state.
    * [`expectimax/`](game2048/expectimax): a powerful ExpectiMax agent by [here](https://github.com/nneonneo/2048-ai).
* [`explore.ipynb`](explore.ipynb): introduce how to use the `Agent`, `Display` and `Game`.
* [`static/`](static/): frontend assets (based on Vue.js) for web app.
* [`webapp.py`](webapp.py): run the web app (backend) demo.
* [`evaluate.py`](evaluate.py): evaluate your self-defined agent.
* [`Net.py`](net.py): cnn model.
* [`train.py`](train.py): start use the tranning and test data to train model.
* [`training_data.py`](evaluate.py): use ExpectiMax agent to play and save the board and the correspongding direction
* [`dataloader.py`](evaluate.py): load tranning and test data.

# To compile the pre-defined ExpectiMax agent

```bash
cd game2048/expectimax
bash configure
make
```
