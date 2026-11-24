# Purple Forensics AI

Infraestructura defensiva de ciberseguridad con análisis forense, enfoque Purple Team.

## Estructura
| Módulo | Qué hace |
|---|---|
| `pfai/models/lvgan.py` | Latent-Variable GAN: L_GAN + L_Dislike + L_prior; score de anomalía; muestras sintéticas `xp` |
| `pfai/models/moe.py` | Mixture of Experts (top-k gating + balanceo de carga) para clasificar familia de ataque |
| `pfai/privacy/synth.py` | CTGAN y Gaussian Copula (SDV) + `DPTabGAN` (DP-SGD con Opacus) |
| `pfai/privacy/audit.py` | KS, error de correlación, TSTR (F1/precisión/recall), AUC de membership-inference, duplicados |
| `pfai/agents/orchestrator.py` | Bus asíncrono + agentes Detector, Triage, Forensic, Sandbox, Report |
| `pfai/tools/forensics.py` | tshark (Wireshark), nmap con `Scope` obligatorio, Sleuth Kit (motor de Autopsy) |
| `pfai/sandbox/sandbox.py` | Sandbox Docker/gVisor sin red + punto de extensión `VMaxSandbox` |
| `pfai/integrity/ledger.py` | Cadena de hashes de evidencias + anclaje opcional en Stellar (testnet por defecto) |
| `pfai/integrations/openclaw.py` | Cliente HTTP genérico (ajustar al API real) |

## Uso
```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/demo.py     # entrena LVGAN + MoE y corre los agentes
PYTHONPATH=. pytest tests
```

## Estado y límites (honestos)
- Probado aquí: ledger, guardas de alcance y bus de agentes (con modelos simulados). El código con PyTorch/Opacus/SDV
  no se pudo ejecutar en este entorno por falta de espacio: espera algún ajuste en el primer run.
- CTGAN/Gaussian Copula NO dan privacidad diferencial por sí solos; la garantía (ε,δ) viene de `DPTabGAN`.
- Autopsy es GUI: se usa Sleuth Kit por CLI y puedes abrir las mismas imágenes en Autopsy.
- `VMaxSandbox` y `OpenClawClient` son stubs: no conozco sus APIs. Docker estándar NO es suficiente para detonar malware;
  usa gVisor (`runtime="runsc"`), Kata, Firecracker o una VM aislada.
- Nmap solo corre contra CIDRs autorizados (`Scope`). Usa el sistema únicamente sobre infraestructura propia o con permiso escrito.
- Stellar solo ancla el hash raíz; usa testnet hasta validar.
- Los datos sintéticos del demo son de juguete; para resultados reales entrena con CIC-IDS2017, UNSW-NB15 o tus propios logs.
