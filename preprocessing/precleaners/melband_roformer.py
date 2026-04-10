import pathlib
import subprocess
import sys
from typing import List


class MelBandRoformerPrecleaner:
    """
    Run an external Music-Source-Separation-Training inference script once per raw dataset
    and reuse the cleaned waveforms for binarization.
    """

    def __init__(self, precleaner_args=None):
        if precleaner_args is None:
            precleaner_args = {}

        self.python_bin = str(precleaner_args.get('python_bin', sys.executable))
        self.script_path = pathlib.Path(precleaner_args.get('script_path', ''))
        self.model_type = str(precleaner_args.get('model_type', 'mel_band_roformer'))
        self.config_path = pathlib.Path(precleaner_args.get('config_path', ''))
        self.ckpt_path = pathlib.Path(precleaner_args.get('ckpt_path', ''))

        self.input_subdir = str(precleaner_args.get('input_subdir', 'wavs'))
        self.output_subdir = str(precleaner_args.get('output_subdir', 'wavs_precleaned'))
        self.filename_template = str(precleaner_args.get('filename_template', '{file_name}'))
        self.pcm_type = str(precleaner_args.get('pcm_type', 'FLOAT'))

        self.force_cpu = bool(precleaner_args.get('force_cpu', False))
        self.disable_detailed_pbar = bool(precleaner_args.get('disable_detailed_pbar', True))
        self.overwrite = bool(precleaner_args.get('overwrite', False))

        extra_cli_args = precleaner_args.get('extra_cli_args', [])
        if isinstance(extra_cli_args, str):
            extra_cli_args = [extra_cli_args]
        self.extra_cli_args = [str(arg) for arg in extra_cli_args]

        self._prepared_dirs = {}
        self._validate_required_paths()

    def _validate_required_paths(self):
        if not self.script_path.exists():
            raise FileNotFoundError(
                f'Precleaner script is not found: {self.script_path}. '
                f'Please set precleaner_args.script_path to the external inference.py path.'
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f'Precleaner config is not found: {self.config_path}. '
                f'Please set precleaner_args.config_path to the MelBand-Roformer config yaml path.'
            )
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f'Precleaner checkpoint is not found: {self.ckpt_path}. '
                f'Please set precleaner_args.ckpt_path to the denoisedebleed.ckpt path.'
            )

    @staticmethod
    def _collect_audio_files(folder: pathlib.Path):
        return sorted([
            p
            for ext in ('*.wav', '*.flac')
            for p in folder.rglob(ext)
        ])

    def _is_output_ready(self, input_files: List[pathlib.Path], output_dir: pathlib.Path):
        if not output_dir.exists():
            return False
        output_files = self._collect_audio_files(output_dir)
        if len(output_files) == 0:
            return False

        input_stems = {p.stem for p in input_files}
        output_stems = {p.stem for p in output_files}
        return input_stems.issubset(output_stems)

    def _run_inference(self, input_dir: pathlib.Path, output_dir: pathlib.Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_bin,
            str(self.script_path),
            '--model_type', self.model_type,
            '--config_path', str(self.config_path),
            '--start_check_point', str(self.ckpt_path),
            '--input_folder', str(input_dir),
            '--store_dir', str(output_dir),
            '--filename_template', self.filename_template,
            '--pcm_type', self.pcm_type,
        ]
        if self.force_cpu:
            cmd.append('--force_cpu')
        if self.disable_detailed_pbar:
            cmd.append('--disable_detailed_pbar')
        cmd.extend(self.extra_cli_args)

        print(f"| preclean command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    def prepare(self, raw_data_dir: pathlib.Path):
        raw_data_dir = pathlib.Path(raw_data_dir).resolve()
        if raw_data_dir in self._prepared_dirs:
            return self._prepared_dirs[raw_data_dir]

        input_dir = raw_data_dir / self.input_subdir
        if not input_dir.exists():
            raise FileNotFoundError(f'Input wav folder is not found: {input_dir}')
        input_files = self._collect_audio_files(input_dir)
        if len(input_files) == 0:
            raise FileNotFoundError(f'No waveform files found in: {input_dir}')

        output_dir = raw_data_dir / self.output_subdir
        if self.overwrite or not self._is_output_ready(input_files, output_dir):
            print(f"| precleaning dataset: {raw_data_dir}")
            self._run_inference(input_dir, output_dir)
        else:
            print(f"| precleaning skipped (already ready): {output_dir}")

        self._prepared_dirs[raw_data_dir] = output_dir
        return output_dir
