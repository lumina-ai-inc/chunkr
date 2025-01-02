This folder contains a simplified/edited implementation of VGT for document layout analysis.

VGT is a two-stream multi-modal Vision Grid Transformer for document layout analysis, in which Grid Transformer (GiT) is proposed and pre-trained for 2D token-level and segment-level semantic understanding. By fully leveraging multi-modal information and exploiting pre-training techniques to learn better representation, VGT achieves highly competitive scores in the DLA task, and significantly outperforms the previous state-of-the-arts.


## Paper
* [ICCV 2023]
* [Arxiv](https://arxiv.org/abs/2308.14978)

If you would like to hit the server, refer to `services/tests/hit_vgt_server.py`.

```bash
sudo docker compose build
sudo docker compose up
```


