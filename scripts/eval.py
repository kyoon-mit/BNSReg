# TEMPORARY SCRIPT

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import corner
import matplotlib.pyplot as plt
import torch
import wandb
from data_regression import LitBNSDataModule

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
            'dec': np.linspace(-np.pi/2, np.pi/2, 41),
            'phi': np.linspace(0, np.pi, 41),
            'snr': np.linspace(0, 100, 41)
        }
        # iterate over a static copy so we can safely add new keys
        for key, item in list(self._bins.items()):
            self._bins[f'{key}_normalized'] = np.linspace(-3, 3, 30)

    def load_config(self, config_path: str) -> tuple[dict, dict, list]:
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.data_init_args = cfg.get('data', {}).get('init_args', {})
        self.model_init_args = cfg.get('model', {}).get('init_args', {})
        self.variables = self.data_init_args.get('variables')
        self.loss = self.model_init_args.get('loss')
        if not isinstance(self.variables, list):
            self.variables = list(self.variables)
            self.data_init_args['variables'] = self.variables
        print(f'Variables: {self.variables}.')
        self.data_init_args['include_snr'] = True
        
        # Get wandb config info
        trainer_config = cfg.get('trainer', {})
        logger_config = trainer_config.get('logger', {}).get('init_args', {})
        self.wandb_project = logger_config.get('project')
        self.wandb_id = logger_config.get('id')

    def load_model(self):
        from model_mse import LitS4ModelMeanOnly
        self.model = LitS4ModelMeanOnly(**self.model_init_args)
        checkpoint = torch.load(self.checkpoint_path, weights_only=True,
                                map_location=torch.device('cpu' if self.compute_on_cpu else 'cuda'))
        self.model.load_state_dict(checkpoint['state_dict'])
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

    def compute_vals(self):
        if Path(self.csv_path).exists():
            return
        self.load_model()
        bns_data_module = LitBNSDataModule(**self.data_init_args)
        bns_data_module.setup('test')
        test_data_loader = bns_data_module.test_dataloader()

        # Placeholders for values
        pred_dict, truth_dict = self.dinit(['pred', 'truth'], self.variables)
        snr = []
        if not self.loss=='MSELoss':
            pred_sigma_dict = self.dinit(['sigma'], self.variables)
        return_dict = dict()

        for batch in test_data_loader:
            h1, l1, params, idx = batch
            device = h1.device
            inputs = torch.stack([h1.to(device), l1.to(device)], dim=2)
            truths = torch.stack([params[v] for v in self.variables], dim=1)

            with torch.no_grad():
                preds = self.model(inputs)

            # Move samples to CPU if required
            if self.compute_on_cpu:
                preds = preds.cpu()
                truths = truths.cpu()
            
            for i in range(len(self.variables)):
                p = self.variables[i]
                pred_dict[f'pred_{p}'].extend(preds[:, i].tolist())
                truth_dict[f'truth_{p}'].extend(truths[:, i].tolist())
                if not self.loss=='MSELoss':
                    pred_sigma_dict[f'sigma_{p}'].extend(preds[:, i+len(self.variables)].tolist())
            
            # Add SNR value
            snr.extend(params['snr'])

        return_dict.update(self.maketensor(pred_dict))
        return_dict.update(self.maketensor(truth_dict))
        return_dict['snr'] = torch.tensor(snr)
        # TODO: implement for other losses
        if not self.loss=='MSELoss':
            return_dict.update(self.makesqrttensor(pred_sigma_dict))
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
        fig.savefig(save_to, bbox_inches='tigsht')
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
    checkpoint_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/chirp_mass_test/config_chirp_mass_d256_l8_48424918/checkpoints/bns-ckpt-epoch=208.ckpt'
    config_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/ssm_regression/tmp_lightning/chirp_mass_test/config_chirp_mass_d256_l8_48424918/config/config_20251130135327.yaml'
    save_path = '/n/holystore01/LABS/iaifi_lab/Lab/kyoon/BNSReg/outputs'
    bns_eval = BNSEval(config_path=config_path, checkpoint_path=checkpoint_path,
                       save_path=save_path, save_suffix='test', compute_on_cpu=False)
    bns_eval.compute_vals()
    bns_eval.plot()
    return

if __name__ == '__main__':
    main()