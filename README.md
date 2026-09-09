# CSI300 Seed-42 Ablation Reproduction Bundle

Private reproducibility bundle for the V100 32 GB ablation experiment.

- `ablation_seed42_code.tar.gz`: model, runner, frozen protocol, tests, and environment files
- `dataset_parts/ablation_seed42_dataset.tar.gz.part*`: dataset-only mount bundle split for connector transport
- `SHA256SUMS.txt`: archive integrity checks

The experiment is limited to random seed 42. It runs six configurations over
five frozen windows (30 runs) and does not make multi-seed stability claims.

Extract the code archive, mount or extract the dataset archive, then follow the
included `README.md`. Keep the dataset read-only when mounted.

Reassemble the dataset on Linux:

```bash
cat dataset_parts/ablation_seed42_dataset.tar.gz.part* > ablation_seed42_dataset.tar.gz
sha256sum -c SHA256SUMS.txt
tar -xzf ablation_seed42_dataset.tar.gz
```

On PowerShell:

```powershell
$parts = Get-ChildItem dataset_parts/ablation_seed42_dataset.tar.gz.part* | Sort-Object Name
$out = [IO.File]::Create('ablation_seed42_dataset.tar.gz')
try { foreach ($part in $parts) { $bytes = [IO.File]::ReadAllBytes($part.FullName); $out.Write($bytes, 0, $bytes.Length) } } finally { $out.Dispose() }
Get-FileHash -Algorithm SHA256 ablation_seed42_dataset.tar.gz
```

