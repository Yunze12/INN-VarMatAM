# Effectiveness of Invertible Neural Network in Variable Material 3D Printing

This repository contains the code and dataset used in the paper "Effectiveness of Invertible Neural Network in Variable Material 3D Printing: Application to Screw-Based Material Extrusion" by Yunze Wang, Beining Zhang, Siwei Lu, Chuncheng Yang, Ling Wang, Jiankang He, Changning Sun, and Dichen Li.

## Repository Structure

* `data/`: Contains the experimental dataset used for training and testing the model.

* `results/`: Contains the results of the training process, including model checkpoints and evaluation metrics.

* `src/`: Contains all the source code used in the project, including data processing, model training, and evaluation scripts.

## Getting Started

### Prerequisites

* Python 3.8 or higher

* PyTorch 1.10 or higher

* Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Yunze12/INN-VarMatAM.git
   cd INN-VarMatAM
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Dataset

The dataset is located in the `data/` folder. If you need to download the dataset, you can place it in this folder.

### Training

To train the model, run the following command:

```bash
python src/train.py
```

This script will train the INN model using the dataset in the `data/` folder and save the results to the `results/` folder.

### Evaluation

To evaluate the trained model, run the following command:

```bash
python src/eval_forward.py
```

This script will load the trained model from the `results/` folder, evaluate its performance on the test dataset, and print the evaluation metrics.

### Optimization

To perform optimization using the trained model, run the following command:

```bash
python src/optimization.py
```

This script will use the trained model to perform inverse optimization and generate posterior samples for the given target outputs. The localization and optimization process is inspired by the work of MatDesINNe, where invertible neural networks are used for inverse materials design.

## Reference

For further reading and related work, please refer to the following references:

* Fung, V., Zhang, J., Hu, G., Ganesh, P., & Sumpter, B. G. (2021). Inverse design of two-dimensional materials with invertible neural networks. *npj Computational Materials*, 7(1), 1-9. [Link](https://github.com/jxzhangjhu/MatDesINNe/tree/main)

* Ardizzone, L., Kruse, J., Wirkert, S., Rahner, D., Pellegrini, E. W., Klessen, R. S., Maier-Hein, L., Rother, C., & Köthe, U. (2019). Analyzing Inverse Problems with Invertible Neural Networks. *arXiv preprint arXiv:1808.04730*. [Link](https://arxiv.org/abs/1808.04730)

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{wang2025effectiveness,
  title={Effectiveness of Invertible Neural Network in Variable Material 3D Printing: Application to Screw-Based Material Extrusion},
  author={Wang, Yunze and Zhang, Beining and Lu, Siwei and Yang, Chuncheng and Wang, Ling and He, Jiankang and Sun, Changning and Li, Dichen},
  journal={Additive Manufacturing Frontiers},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Special thanks to the authors of MatDesINNe for their work on inverse materials design via invertible neural networks, which inspired the localization and optimization process in this repository.

## Contact

For any questions or inquiries, please contact: Yunze Wang (wangyunze@stu.xjtu.edu.cn)
