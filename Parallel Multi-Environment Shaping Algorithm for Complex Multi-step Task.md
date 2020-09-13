# RL Agent For PySC2 Environment

[![MoveToBeacon](https://user-images.githubusercontent.com/195271/37241507-0d7418c2-2463-11e8-936c-18d08a81d2eb.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralShards](https://user-images.githubusercontent.com/195271/37241785-b8bd0b04-2467-11e8-9ff3-e4335a7c20ee.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatRoaches](https://user-images.githubusercontent.com/195271/37241527-32a43ffa-2463-11e8-8e69-c39a8532c4ce.gif)](https://youtu.be/gEyBzcPU5-w)
[![DefeatZerglingsAndBanelings](https://user-images.githubusercontent.com/195271/37241531-39f186e6-2463-11e8-8aac-79471a545cce.gif)](https://youtu.be/gEyBzcPU5-w)
[![FindAndDefeatZerglings](https://user-images.githubusercontent.com/195271/37241532-3f81fbd6-2463-11e8-8892-907b6acebd04.gif)](https://youtu.be/gEyBzcPU5-w)
[![CollectMineralsAndGas](https://user-images.githubusercontent.com/195271/37241521-29594b48-2463-11e8-8b43-04ad0af6ff3e.gif)](https://youtu.be/gEyBzcPU5-w)
[![BuildMarines](https://user-images.githubusercontent.com/195271/37241515-1a2a5c8e-2463-11e8-8ac4-588d7826e374.gif)](https://youtu.be/gEyBzcPU5-w)


## Introduction
This repo is for this paper [Parallel Multi-Environment Shaping Algorithm for Complex Multi-step Task](https://www.sciencedirect.com/science/article/pii/S092523122030655X?via%3Dihub). In this paper, we propose a novel algorithm called Paralleled Multi-Environment Shaping algorithm (PMES), where several sub-environments are built based on human knowledge to make the agent aware of the importance of intermediate steps, each of which corresponds to each key intermediate steps. 

## Running
We add some new map for task BuildMarines. To use to the new training with new maps, **copy the killers_map directory to StarcraftII/Maps/**

Note:
if you set ``--maps``, then ``--envs`` and ``map`` will be ignored.

* To train a PMES agent, execute `python main.py --gpu $GPU --work_dir *** --run_id 1 --sz 16 --envs 16 --render 0 --updates 50000 --lr 0.00001 --vf_coef 0.25 --ent_coef 0.00009 --discount 0.99 --clip_grads 1 --save_interval 1000 --maps '{"BuildMarines_15min": 10, "BuildMarinesA_4-6min_RP_new1": 1, "BuildMarinesB_4-6min_RP_new1": 3, "BuildMarinesC_4-6min_ARP_newx": 2}' --num_snapshot 5 --optimizer adam --beta1 0.9 --beta2 0.999 --step_mul 8`.

* To train a shaping agent, execute `python main.py --gpu $GPU --work_dir ** --run_id 2 --sz 16 --envs 16 --render 0 --updates 500000 --lr 0.00001 --vf_coef 0.25 --ent_coef 0.00009 --discount 0.99 --clip_grads 1 --save_interval 1000 --map BuildMarines_15min_shaping_ARP_111 --num_snapshot 1000 --optimizer adam --beta1 0.9 --beta2 0.999 --step_mul 8"`

* To train a curriculum learning agent, execute `python main.py --gpu $GPU --work_dir ** --run_id 1 --sz 16 --envs 16 --render 0 --updates 500000 --lr 0.00001 --vf_coef 0.25 --ent_coef 0.00009 --discount 0.99 --clip_grads 1 --save_interval 1000 --map BuildMarinesA_4-6min_RP_new1 --num_snapshot 5 --optimizer adam --beta1 0.9 --beta2 0.999 --step_mul 8`, denoted as "agent_A"; then initialize the new agent using "agent_A", and train it under the SubEnv_B, denoted as "agent_B"; as in this way, train the agent under SubEnv_C, OrigEnv in order. 

* To train a PLAID agent, like the curriculum learning agent, get the "agent_A" and "agent_B"; then, set '--distill' is True, and set the path of "agent_A" and "agent_B", to learn a new agent, denoted as "agent_AB"; then train this agent under SubEnv_C to get the "agent_C', distill "agent_AB" and "agent_C" to get "agent_ABC". Lastly, we train the agent under the original environment OrigEnv based on "agent_ABC". Detailed steps can refer to our [paper](https://www.sciencedirect.com/science/article/pii/S092523122030655X?via%3Dihub). 

* To resume training from last checkpoint, specify `--restore` flag
* To run in inference mode, specify `--test` flag
* To change number of rendered environments, specify `--render=` flag
* To change state/action space, specify path to a json config with `--cfg_path=`. The configuration with reduced feature space used to achieve some of the results above is:

```json
{
  "feats": {
    "screen": ["visibility_map", "player_relative", "unit_type", "selected", "unit_hit_points_ratio", "unit_density"],
    "minimap": ["visibility_map", "camera", "player_relative", "selected"],
    "non_spatial": ["player", "available_actions"]
  }
}
```

### Requirements

* Python 3.x
* Tensorflow >= 1.3
* PySC2 **1.2** [with action spec fix](https://github.com/deepmind/pysc2/pull/105)

Good GPU and CPU are recommended, especially for full state/action space.

## Video
For the trained agents of different algorithms, please watch this [Supplementary materials](https://ars.els-cdn.com/content/image/1-s2.0-S092523122030655X-mmc1.mp4). 

## Acknowledgements

Firstly, many thanks to [Roman Ring](https://github.com/inoryy). We make some improvement based on his bachelor thesis, and then proposed the [PMES algorithm](https://www.sciencedirect.com/science/article/pii/S092523122030655X?via%3Dihub). Here, we would like to thank the members of The Killers team and their instructors Lisen Mu and Jingchu Liu in DeeCamp 2018 for the insightful discussions. The group members of Killers have Zhizhong Li, Cong Ma, Hongyi Guo, Jiamin He, Bin Hu, Yiquan Lin, Yinmin Zhang, Dan Lin, Mincai Lai, Gangyi Lin. Also, we would like to thank Ke Bai for the comments.


## Related literature

[1] [Parallel Multi-Environment Shaping Algorithm for Complex Multi-step Task](https://www.sciencedirect.com/science/article/pii/S092523122030655X?via%3Dihub)

[2] [Replicating deepmind starcraft ii reinforcement learning benchmark with actor-critic methods](http://hdl.handle.net/10062/61039)

For more literature, please refer to the References part of our [paper](https://www.sciencedirect.com/science/article/pii/S092523122030655X?via%3Dihub). 

## Citation section

Ma, C., Li, Z., Lin, D. and Zhang, J., 2020. Parallel Multi-Environment Shaping Algorithm for Complex Multi-step Task. Neurocomputing, 402, pp. 323--335.

* Bibtex
@article{DBLP:journals/ijon/MaLLZ20,
  author    = {Cong Ma and
               Zhizhong Li and
               Dahua Lin and
               Jiangshe Zhang},
  title     = {Parallel Multi-Environment Shaping Algorithm for Complex Multi-step
               Task},
  journal   = {Neurocomputing},
  volume    = {402},
  pages     = {323--335},
  year      = {2020},
  url       = {https://doi.org/10.1016/j.neucom.2020.04.070},
  doi       = {10.1016/j.neucom.2020.04.070},
  timestamp = {Mon, 15 Jun 2020 16:52:59 +0200},
  biburl    = {https://dblp.org/rec/journals/ijon/MaLLZ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
