import hydra
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from omegaconf import OmegaConf
from sklearn import manifold
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.tsne import visualize_tsne_points
from utils.utils import set_seed

color = sns.color_palette("colorblind")

dataset_to_color = {
    'waymo': color[2],
    'nuplan': color[0],
    'av2': color[3],
    'nuscenes': color[4],
}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def tsne(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    model = build_model(cfg)
    train_set = build_dataset(cfg)

    train_loader = DataLoader(
        train_set, batch_size=4, num_workers=cfg.load_num_workers, drop_last=False, shuffle=False,
        collate_fn=train_set.collate_fn)

    trainer = pl.Trainer(
        inference_mode=True,
        devices=1,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
    )

    predict = trainer.predict(model=model, dataloaders=train_loader)  # , ckpt_path=cfg.ckpt_path)

    # concatentate all the predictions
    scene_emb = []
    dataset_name = []
    for i in range(len(predict)):
        scene_emb.append(predict[i][0]['scene_emb'])
        dataset_name.append(predict[i][0]['dataset_name'])

    embeds = torch.cat(scene_emb, dim=0).cpu().numpy()
    dataset_list = np.concatenate(dataset_name, axis=0)

    datasize = embeds.shape[0]
    point_num = datasize

    vis_num = int(point_num * 0.1)

    # Draw a 3D TSNE
    tsne = manifold.TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        n_jobs=-1,
    )

    c_list = [dataset_to_color[c] for c in dataset_list]
    Y = tsne.fit_transform(embeds)

    ax = visualize_tsne_points(Y, c_list)
    tsne_points = wandb.Image(ax)

    # random select vis_num idx from point_num
    rand_indx = list(range(point_num))
    np.random.shuffle(rand_indx)
    rand_indx = rand_indx[:vis_num]

    c_list = np.array(c_list)[rand_indx]
    Y = Y[rand_indx]
    # image_lsit = []
    # for idx in rand_indx:
    #     ax = draw_scene(all_results['ego_full'][idx],all_results['other_full'][idx],all_results['map'][idx])
    #     # save
    #     os.makedirs('./img', exist_ok=True)
    #     save_path = os.path.join('./img', f'{idx}.png')
    #     ax.figure.savefig(save_path)
    #     image_lsit.append(save_path)

    # tsne_image = wandb.Image(visualize_tsne_images(Y[:, 0], Y[:, 1], image_lsit, c_list,max_image_size=500))

    wandb.init(project='motionnet_vis', name=cfg.exp_name)
    wandb.log({"tsne_points": tsne_points})  # ,'tsne_image':tsne_image})


if __name__ == "__main__":
    tsne()
