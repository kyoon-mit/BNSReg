# TEMPORARY SCRIPT

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
import torch
import wandb
import importlib
from BNSReg.dataloader.regression_loader import LitBNSDataRegression
from BNSReg.core.config import BNSDataModuleRegressionConfig

class BNSEval():
    def __init__(
            self,
            config_path: str,
            checkpoint_path: str,
            save_path: Path | str,
            save_suffix: str,
            compute_on_cpu: bool=False):
        self.load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.compute_on_cpu = compute_on_cpu
        self.save_path = Path(save_path)
        self.csv_path = self.save_path / 'bns_eval.csv'
        self.save_suffix = save_suffix
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._var_map = {
            'mass_1': 'm_1',
            'mass_2': 'm_2',
            's1z': 'S_{1,z}',
            's2z': 'S_{2,z}',
            'distance': r'\mathrm{distance}',
            'phic': r'\phi_c',
            'dec': r'\mathrm{declination}',
            'phi': r'\phi',
            'inclination': r'\mathrm{inclination}',
            'chirp_mass': r'\mathcal{M}',
            'total_mass': 'M',
            'mass_ratio': 'q',
            'snr': 'snr',
            'mass_1_normalized': r'\tilde{m}_1',
            'mass_2_normalized': r'\tilde{m}_2',
            's1z_normalized': r'\tilde{S}_{1,z}',
            's2z_normalized': r'\tilde{S}_{2,z}',
            'distance_normalized': r'\mathrm{distance_normalized}',
            'phic_normalized': r'\tilde{\phi}_c',
            'dec_normalized': r'\mathrm{declination_normalized}',
            'phi_normalized': r'\tilde{\phi}',
            'inclination_normalized': r'\mathrm{inclination_normalized}',
            'chirp_mass_normalized': r'\tilde{\mathcal{M}}',
            'total_mass_normalized': r'\tilde{M}',
            'mass_ratio_normalized': r'\tilde{q}',
            'snr_normalized': 'snr_normalized'
        }
        self._bins = {
            'chirp_mass': np.linspace(0.5, 2.5, 41),
            'mass_ratio': np.linspace(0.0, 1.0, 41),
            'dec': np.linspace(-np.pi/2, np.pi/2, 41),
            'phi': np.linspace(-np.pi, np.pi, 41),
            'snr': np.linspace(0, 100, 41),
        }
        # iterate over a static copy so we can safely add new keys
        for key, item in list(self._bins.items()):
            self._bins[f'{key}_normalized'] = np.linspace(-3, 3, 30)

    def load_config(self, config_path: str) -> None:
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.data_cfg_dict = cfg['data']['init_args']['data_cfg']
        self.model_class_path = cfg['model']['class_path']
        self.target_variables = self.data_cfg_dict['target_variables']
        self.observed_variables = self.data_cfg_dict.get('observed_variables', [])
        print(f'Target variables: {self.target_variables}.')
        print(f'Observed variables: {self.observed_variables}.')

        trainer_config = cfg.get('trainer', {})
        logger_config = trainer_config.get('logger', {}).get('init_args', {})
        self.wandb_project = logger_config.get('project')
        self.wandb_id = logger_config.get('id')

    def load_model(self):
        module_path, class_name = self.model_class_path.rsplit('.', 1)
        ModelClass = getattr(importlib.import_module(module_path), class_name)
        device = 'cpu' if (self.compute_on_cpu or not torch.cuda.is_available()) else 'cuda'
        self.model = ModelClass.load_from_checkpoint(
            self.checkpoint_path, map_location=device, weights_only=False)
        self.model.eval()
        
    def maketensor(self, d: dict) -> dict:
        return {k: torch.tensor(v) for k, v in d.items()}

    def makesqrttensor(self, d: dict) -> dict:
        return {k: torch.sqrt(torch.tensor(v)) for k, v in d.items()}

    def dinit(self, prefix: list, variables: list) -> tuple[dict] | dict:
        out = []
        for p in prefix:
            d = {f'{p}_{var}': [] for var in variables}
            out.append(d)
        return out[0] if len(out) == 1 else tuple(out)
        
    def dump_to_csv(self, outputs: dict, csv_path: str) -> pd.DataFrame:
        df = pd.DataFrame(outputs)
        df.to_csv(csv_path, index=False)
        print(f'Dumped results to {csv_path}')
        return df

    def _unnormalize(self, arr: torch.Tensor, variables: list) -> torch.Tensor:
        lo, hi = self._normalize_range
        out = arr.clone().float()
        for i, var in enumerate(variables):
            if var in self._var_scales:
                vmin, vmax = self._var_scales[var]
                out[:, i] = vmin + (vmax - vmin) * (arr[:, i] - lo) / (hi - lo)
        return out

    def compute_vals(self):
        if Path(self.csv_path).exists():
            return
        self.load_model()
        data_cfg = BNSDataModuleRegressionConfig(**self.data_cfg_dict)
        data_module = LitBNSDataRegression(data_cfg=data_cfg)
        data_module.setup('test')
        test_loader = data_module.test_dataloader()

        # Set up normalization: var_scales always derived from train_file
        self._var_scales = dict(getattr(data_module.test_dataset, 'var_scales', {}))
        self._normalize_range = tuple(self.data_cfg_dict.get('normalize_range', [0.0, 1.0]))
        do_unnorm = bool(self._var_scales) and self.data_cfg_dict.get('normalize_variables', False)

        n_targets = len(self.target_variables)
        pred_dict, truth_dict = self.dinit(['pred', 'truth'], self.target_variables)
        snr = []
        has_snr = 'snr' in self.observed_variables
        snr_idx = self.observed_variables.index('snr') if has_snr else None
        return_dict = dict()

        device = next(self.model.parameters()).device
        for batch in test_loader:
            X_sequence, y_target, z_observed = batch
            X_sequence = X_sequence.transpose(2, 1).to(device)  # (B, L, d_input)

            with torch.no_grad():
                preds = self.model(X_sequence)  # (B, d_output)

            preds = preds.cpu()
            # first n_targets columns are means; remaining (if any) are variances
            means = preds[:, :n_targets]

            if do_unnorm:
                means    = self._unnormalize(means,    self.target_variables)
                y_target = self._unnormalize(y_target, self.target_variables)

            for i, var in enumerate(self.target_variables):
                pred_dict[f'pred_{var}'].extend(means[:, i].tolist())
                truth_dict[f'truth_{var}'].extend(y_target[:, i].tolist())

            if has_snr:
                snr_col = z_observed[:, snr_idx:snr_idx + 1]
                if do_unnorm and 'snr' in self._var_scales:
                    snr_col = self._unnormalize(snr_col, ['snr'])
                snr.extend(snr_col[:, 0].tolist())

        return_dict.update(self.maketensor(pred_dict))
        return_dict.update(self.maketensor(truth_dict))
        if has_snr:
            return_dict['snr'] = torch.tensor(snr)
        self.dump_to_csv(return_dict, csv_path=self.csv_path)
        return

    def plot(self):
        if not Path(self.csv_path).exists:
            raise RuntimeError('Please run compute_vals first.')
        df = pd.read_csv(self.csv_path)

        # Get variable names (everything after the prefix)
        pred_cols = [col for col in df.columns if col.startswith('pred_')]
        truth_cols = [col for col in df.columns if col.startswith('truth_')]
        assert len(pred_cols)==len(truth_cols)
        variables = [col.removeprefix('pred_') for col in pred_cols]

        # Create labels
        tex_variables = [self._var_map[var] for var in variables]
        label_preds = [fr'$\hat{{{tvar}}}$' for tvar in tex_variables]
        label_truths = [fr'${tvar}$' for tvar in tex_variables]
        label_residuals = [fr'$\hat{{{tvar}}} - {tvar}$' for tvar in tex_variables]
        label_z_scores  = [fr'$(\hat{{{tvar}}} - {tvar})/\hat\sigma_{{{tvar}}}$' for tvar in tex_variables]
        label_sigmas = [fr'$\hat\sigma_{tvar}' for tvar in tex_variables]

        # Get predictions and truths
        preds = np.stack([df[p] for p in pred_cols], axis=1)
        truths = np.stack([df[t] for t in truth_cols], axis=1)
        snr = np.array(df['snr'])

        # Calculate residuals
        residuals = preds - truths

        # Calculate L1 (mean absolute error) and L2 (root mean square error) distances
        l1_distances = np.mean(np.abs(residuals), axis=0)
        l2_distances = np.sqrt(np.mean(residuals**2, axis=0))
        
        # Log to wandb if available
        # if hasattr(self, 'wandb_id') and self.wandb_id:
        #     try:
        #         # Resume the existing wandb run
        #         wandb.init(project=self.wandb_project, id=self.wandb_id, resume='must')
                
        #         # Log the metrics
        #         metrics = {}
        #         for var, l1_dist, l2_dist in zip(variables, l1_distances, l2_distances):
        #             metrics[f'eval/l1_mae_{var}'] = l1_dist
        #             metrics[f'eval/l2_rmse_{var}'] = l2_dist
        #         wandb.log(metrics)
                
        #         # Also log the corner plots and histograms
        #         for var in variables:
        #             if (self.save_path / f'hist_{var}.png').exists():
        #                 wandb.log({f'eval/hist_{var}': wandb.Image(str(self.save_path / f'hist_{var}.png'))})
        #         if (self.save_path / f'residuals_{self.save_suffix}.png').exists():
        #             wandb.log({f'eval/residuals': wandb.Image(str(self.save_path / f'residuals_{self.save_suffix}.png'))})

        #         # Upload CSV as a wandb Artifact (dataset)
        #         try:
        #             if Path(self.csv_path).exists():
        #                 artifact_name = f'bns_eval_{self.save_suffix}'
        #                 art = wandb.Artifact(artifact_name, type='dataset')
        #                 art.add_file(str(self.csv_path))
        #                 wandb.log_artifact(art)
        #         except Exception as e2:
        #             print(f'Warning: failed to upload CSV artifact to wandb: {e2}')
                
        #         wandb.finish()
        #     except Exception as e:
        #         print(f'Warning: Failed to log to wandb: {e}')
        
        # Check if sigma columns exist and stack normalized residuals
        sigma_cols = [f'sigma_{var}' for var in variables]
        sigma_exists = all(col in df.columns for col in sigma_cols)
        if sigma_exists:
            sigmas = np.stack([df[s] for s in sigma_cols], axis=1)
            z_scores = np.divide(residuals, sigmas,
                                 out=np.full_like(residuals, np.nan),
                                 where=(sigmas != 0))
        
        self._plot_corner(
            title=f'Residuals',
            fig_name=f'residuals_{self.save_suffix}',
            data=residuals,
            labels=label_residuals,
            color='purple')

        self._plot_hists(truths=truths, preds=preds, snr=snr,
                         label_truths=label_truths, label_preds=label_preds,
                         variables=variables)

        if sigma_exists:
            self._plot_corner(suptitle=f'Z Scores',
            fig_name=f'z_scores_{self.save_suffix}',
            data=z_scores,
            labels=label_z_scores,
            color='orange')

            self._plot_corner(suptitle=f'Uncertainties',
            fig_name=f'sigmas_{self.save_suffix}',
            data=sigmas,
            labels=label_sigmas,
            color='blue')
        
        return

    def _plot_corner(self, title: str, fig_name: str | Path, data: np.array, labels: str, color: str, **kwargs) -> None:
        fig = corner.corner(
            data=data,
            labels=labels,
            color=color,
            **kwargs
        )
        fig.suptitle(title, fontsize=12)
        fig.subplots_adjust(top=.87)
        fig_name = Path(fig_name).with_suffix('.png')
        save_to = Path(self.save_path) / fig_name
        fig.savefig(save_to, bbox_inches='tight')
        print(f'Saved {save_to}')
        # Attempt to upload the figure to wandb (resume existing run by id)
        if hasattr(self, 'wandb_id') and self.wandb_id:
            try:
                wandb.init(project=self.wandb_project, id=self.wandb_id, resume='allow')
                wandb.log({f'eval/{save_to.name}': wandb.Image(str(save_to))})
                wandb.finish()
            except Exception as e:
                print(f'Warning: failed to upload {save_to} to wandb: {e}')
        return
    
    def _plot_hists(self, truths: np.array, preds: np.array, snr: np.array,
                    label_truths: list[str], label_preds: list[str], variables: list[str],
                    **kwargs) -> None:
        for i, var in enumerate(variables):
            plt.figure(figsize=(6, 4))
            plt.hist(truths[:, i], bins=self._bins[var], alpha=0.5, color='green', 
                    label=label_truths[i], density=False, histtype='step')
            plt.hist(preds[:, i], bins=self._bins[var], alpha=1.0, color='red', 
                    label=label_preds[i], density=False, histtype='step')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.legend()
            plt.title(f'{var} Distribution')
            plt.tight_layout()
            save_to_1 = Path(self.save_path) / f'hist_{var}.png'
            plt.savefig(save_to_1)
            print(f'Saved to {save_to_1}')
            plt.close()

            # plt.scatter(x=snr, y=(preds[:, i] - truths[:, i])/truths[:, i],
            #            color='blue')
            # plt.xlabel('snr')
            # plt.ylabel(var)
            # plt.title(f'snr vs. {var} residuals')
            # plt.tight_layout()
            # save_to_2 = Path(self.save_path) / f'scatter_snr_{var}.png'
            # plt.savefig(save_to_2)
            # print(f'Saved to {save_to_2}')
            # plt.close()

            plt.figure(figsize=(6, 4))
            plt.hist(preds[:, i] - truths[:, i], np.linspace(-0.5, 0.5, 101), alpha=1.0, color='purple',
                     label=f'{label_preds[i]} - {label_truths[i]}', density=False, histtype='step')
            plt.xlabel('Residual')
            plt.ylabel('Count')
            plt.legend()
            plt.title(f'{var} Residuals')
            plt.tight_layout()
            save_to_3 = Path(self.save_path) / f'hist_resid_{var}.png'
            plt.savefig(save_to_3)
            print(f'Saved to {save_to_3}')
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.hist((preds[:, i] - truths[:, i])/truths[:, i], bins=np.linspace(-0.1, 0.1, 101), alpha=1.0, color='blue',
                     label=f'({label_preds[i]} - {label_truths[i]})/{label_truths[i]}', density=False, histtype='step')
            plt.xlabel('Relative Error')
            plt.ylabel('Count')
            plt.legend()
            plt.title(f'{var} Relative Error')
            plt.tight_layout()
            save_to_4 = Path(self.save_path) / f'hist_rel_err_{var}.png'
            plt.savefig(save_to_4)
            print(f'Saved to {save_to_4}')
            plt.close()
            # Attempt to upload histogram to wandb
            # if hasattr(self, 'wandb_id') and self.wandb_id:
            #     try:
            #         wandb.init(project=self.wandb_project, id=self.wandb_id, resume='allow')
            #         wandb.log({f'eval/hist_{var}': wandb.Image(str(save_to_1))})
            #         wandb.log({f'eval/hist2d_snr_{var}': wandb.Image(str(save_to_2))})
            #         wandb.finish()
            #     except Exception as e:
            #         print(f'Warning: failed to upload histogram(s) to wandb: {e}')

    def trainer_callback(self):
        self.compute_vals()
        self.plot()

def main():
    save_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/SAVED_RESULTS/OLD/s4d_gaussnll_snr_20_30_seq_0_55s/setting4'
    checkpoint_path = f'{save_path}/checkpoints/s4d_gaussnll_ckpt_epoch=120.ckpt'
    config_path = f'{save_path}/s4d_mse_20_30_seq_0_55s_working_setting.yaml'
    bns_eval = BNSEval(config_path=config_path, checkpoint_path=checkpoint_path,
                       save_path=save_path, save_suffix='mse_epoch120', compute_on_cpu=False)
    bns_eval.compute_vals()
    bns_eval.plot()
    return

if __name__ == '__main__':
    main()