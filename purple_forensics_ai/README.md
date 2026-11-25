# Purple Forensics AI (`pfai`)

Infraestructura defensiva y de simulación **Purple Team** orientada al análisis forense, generación privada de datos sintéticos y orquestación multiagente.

## Estructura del paquete

| Módulo | Responsabilidad |
|---|---|
| `pfai/data/synthetic_flows.py` | Generador tabular mixto realista de flujos de red (`benign`, `dos`, `probe`, `bruteforce`, `exfil`) |
| `pfai/preprocess.py` | `TabularEncoder`: codificación mixta bidireccional (`transform` + `inverse_transform`, imputación y límites) |
| `pfai/models/lvgan.py` | `LVGAN`: Encoder-Decoder + Discriminador ($L_{\text{GAN}}$, $L_{\text{Dislike}}$, $L_{\text{prior}}$), `score_components` y `generate_evasion_variants` |
| `pfai/models/moe.py` | `MoEClassifier`: Mixture of Experts (*top-k gating*, balanceo de carga, pesos de clase, `expert_utilization`) |
| `pfai/privacy/synth.py` | `fit_ctgan`, `fit_gaussian_copula`, `DPTabGAN` (Opacus DP-SGD + `PrivacyBudget`) y `benchmark_synthesizers` |
| `pfai/privacy/audit.py` | `fidelity` (KS, corr MAE, TV), `utility_tstr`, `utility_trtr`, `membership_inference_auc`, `distance_to_closest_record`, `duplicate_rate`, `full_report` |
| `pfai/agents/orchestrator.py` | Bus asíncrono + `RedTeamAgent`, `DetectorAgent`, `TriageAgent`, `ForensicAgent`, `SandboxAgent`, `OpenClawAgent`, `ReportAgent` |
| `pfai/tools/forensics.py` | Wrappers sin shell con `Scope` obligatorio para `tshark`, `nmap` y Sleuth Kit (`mmls`/`fls`/`icat`, motor de Autopsy) + parsers estructurados |
| `pfai/sandbox/sandbox.py` | `DockerSandbox` (red deshabilitada, rootfs RO, gVisor/Kata), `VMaxSandbox` (ciclo snapshot -> detonación -> revert) y `StaticFallbackSandbox` |
| `pfai/integrity/ledger.py` | `EvidenceLedger` SHA-256 encadenado (thread-safe) + anclaje y verificación en Stellar (`anchor_stellar`, `verify_stellar_anchor`) |
| `pfai/integrations/openclaw.py` | `OpenClawClient` con reintentos, firma HMAC opcional y sanitización zero-trust de respuestas |
| `pfai/cli.py` | CLI `pfai` (`demo`, `benchmark-privacy`, `capabilities`, `verify-ledger`, `analyze-sample`) |
| `../notebooks/Purple_Forensics_AI_Colab.ipynb` | Notebook ejecutable en Google Colab con los 5 pilares del proyecto |

## Instalación y ejecución

```bash
pip install -e ".[all]"
python scripts/demo.py
pfai capabilities
pytest tests -v
```

### En Google Colab
```python
!pip install -q "git+https://github.com/santy8151/Studia_IA_Reto.git#subdirectory=purple_forensics_ai" opacus sdv stellar-sdk
```
O abre directamente `../notebooks/Purple_Forensics_AI_Colab.ipynb` (desde la raíz del repositorio).

## Garantías, supuestos y límites de seguridad
- **Privacidad Diferencial**: `CTGAN` y `GaussianCopula` **no** ofrecen privacidad diferencial formal por sí solos. La garantía $(\epsilon, \delta)$ proviene de `DPTabGAN` mediante DP-SGD (Opacus), registrada en `PrivacyBudget`.
- **Auditoría empírica (`pfai.privacy.audit`)**: KS, correlación, variación total, TSTR/TRTR, DCR/NNDR, duplicados y `MIA AUC` son diagnósticos empíricos útiles para detectar sobreajuste o memorización; un `MIA AUC` cercano a `0.5` no equivale a una prueba matemática de privacidad.
- **Forense y alcance autorizado (`Scope`)**: `nmap_service_scan` solo acepta direcciones IP literales dentro de los CIDRs autorizados en `Scope` y rechaza nombres de host para impedir inyección o escaneo fuera de alcance. Autopsy se integra vía CLI con The Sleuth Kit (`mmls`, `fls`, `icat`).
- **Sandboxing**: Docker estándar comparte el kernel del host y **no** debe usarse para detonar malware real; úsalo con `runtime="runsc"` (gVisor), Kata/Firecracker o mediante `VMaxSandbox` en una VM dedicada con red aislada. `StaticFallbackSandbox` realiza únicamente inspección estática sin ejecutar la muestra.
- **OpenClaw y LLMs**: toda respuesta externa recibida por `OpenClawClient` se marca y sanitiza como dato no confiable (`untrusted_data=True`), y cualquier planificador dinámico queda restringido por `ALLOWED_TOOLS` y `Scope`.
- **Stellar**: solo se publica en `manage_data` el hash SHA-256 raíz (32 bytes) bajo el nombre `evidence_root`; usa `testnet=True` durante desarrollo y validación.
