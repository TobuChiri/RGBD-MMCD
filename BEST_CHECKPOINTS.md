# Local Verification Checkpoints

Large checkpoint files are not committed to GitHub. The local CoLA workspace used for verification contained the checkpoints below.

## DFormer / CoLA

- `DFormer/checkpoints/NYUD_CD/epoch-292_miou_55.05.pth` (verified mIoU=55.14 on NYUDepthv2)
- `DFormer/checkpoints/NYUD_MD/epoch-289_miou_54.94.pth` (recorded mIoU=54.94)
- `DFormer/checkpoints/NYUDepthv2_DFormer-Base_20251202-002616/epoch-150_miou_44.56.pth` (recorded mIoU=44.56)
- `DFormer/checkpoints/NYUDepthv2_DFormer-Base_20251201-215927/epoch-1_miou_21.6.pth` (recorded mIoU=21.6)

## Sigma_CD

- Local trained Sigma checkpoints were not present in the CoLA workspace.
- VMamba pretrained weights used by the README-reported Sigma variants were present locally but are not committed:
  - `Sigma_CD/pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth`
  - `Sigma_CD/pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth`
  - `Sigma_CD/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth`

Generated from source workspace: `E:\code\CoLA`
