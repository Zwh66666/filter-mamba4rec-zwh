import sys
import logging
import torch
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from mamba4rec import Mamba4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_metrics(metrics):
    sns.set(style="darkgrid")
    plt.figure(figsize=(14, 7))

    for metric in metrics.keys():
        plt.plot(metrics[metric], label=metric)

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics Over Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    config = Config(model=Mamba4Rec, config_file_list=['config.yaml'])
    init_seed(config['seed'], config['reproducibility'])
    
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = Mamba4Rec(config, train_data.dataset).to(config['device']) 
    logger.info(model)
    
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # Initialize metrics storage
    metrics = {'HR': [], 'MRR': [], 'NDCG': []}

    # model training
    for epoch in tqdm(range(config['epochs']), desc="Training"):
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, show_progress=False
        )

        # Evaluate on validation set
        valid_result = trainer.evaluate(valid_data, show_progress=False)
        print("Validation Result:", valid_result)  # 打印验证结果
        # 确保 valid_result 中包含 'recall@10', 'mrr@10', 'ndcg@10' 键
        if 'recall@10' in valid_result:
            metrics['HR'].append(valid_result['recall@10'])  # Example for HR@10
        if 'mrr@10' in valid_result:
            metrics['MRR'].append(valid_result['mrr@10'])    # Example for MRR@10
        if 'ndcg@10' in valid_result:
            metrics['NDCG'].append(valid_result['ndcg@10'])  # Example for NDCG@10

        logger.info(set_color(f"Epoch {epoch+1}", "yellow") + f" - valid: {valid_result}")

    # model evaluation
    test_result = trainer.evaluate(
        test_data, show_progress=config["show_progress"]
    )
    
    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")  

    # Plot metrics
    plot_metrics(metrics)
