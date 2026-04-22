import torch
import pandas as pd
import yaml
from BNSReg.core.config import BNSDataModuleRegressionConfig
from BNSReg.tasks.parameter_estimation.model_s4d_gaussnll import LitModelS4DGaussianNLLLoss
from BNSReg.dataloader.regression_loader import LitBNSDataRegression

global ckpt_path
global config_path
ckpt_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/s4d_gaussnll_15_30_pre_match_filter/two_masses_59s_63s-d64-s64-l4-ckpt4/checkpoints/s4d_gaussnll_var0mseloss_ckpt_epoch=1096.ckpt'
config_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/configs/parameter_estimation/user/mc/s4d_gaussnll_15_30_pre_match_filter/two_masses_59s_63s-d64-s64-l4.yaml'
csv_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs/s4d_gaussnll_15_30_pre_match_filter/two_masses_59s_63s-d64-s64-l4-ckpt4/' + 'background.csv'

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YAML
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Extract model init args
    model_cfg = cfg['model']['init_args']['model_cfg']
    data_cfg = cfg['data']['init_args']['data_cfg']

    # Dial down workers
    data_cfg['num_workers'] = 4
    data_cfg['prefetch_factor'] = 4
    data_cfg['persistent_workers'] = True

    # Explicitly load checkpoint weights only (safe)
    _ = torch.load(ckpt_path, map_location=device, weights_only=True)

    model = LitModelS4DGaussianNLLLoss.load_from_checkpoint(
        ckpt_path,
        model_cfg=model_cfg,
        map_location=device,
    )

    model.to(device)
    model.eval()

    datamodule = LitBNSDataRegression(
        data_cfg=BNSDataModuleRegressionConfig(**data_cfg)
    )
    datamodule.setup(stage='test')
    test_loader = datamodule.test_dataloader()

    out = torch.empty(len(test_loader.dataset), 6)
    i = 0

    with torch.no_grad():
        for batch in test_loader:
            print(i)

            X, y, z = batch
            X = X.transpose(2, 1).to(device)
            y = y.to(device)

            outputs = model(X)
            pred_mean = outputs[:, :2]
            pred_sigma = torch.sqrt(model.var_activation(outputs[:, 2:]))

            # Move back to CPU before writing
            pred_mean = pred_mean.detach().cpu()
            pred_sigma = pred_sigma.detach().cpu()
            y = y.detach().cpu()

            B = y.size(0)
            out[i:i+B, 0] = y[:, :1].squeeze(1)
            out[i:i+B, 1] = y[:, 1:2].squeeze(1)
            out[i:i+B, 2] = pred_mean[:, :1].squeeze(1)
            out[i:i+B, 3] = pred_mean[:, 1:2].squeeze(1)
            out[i:i+B, 4] = pred_sigma[:, :1].squeeze(1)
            out[i:i+B, 5] = pred_sigma[:, 1:2].squeeze(1)

            i += B

    out = out.numpy()

    df = pd.DataFrame(
        out,
        columns=[
            'M_chirp_true', 'M_ratio_true',
            'M_chirp_pred', 'M_ratio_pred',
            'sigma_M_chirp_pred', 'sigma_M_ratio_pred']
    )
    df.to_csv(csv_path, index=False)

if __name__=='__main__':
    main()