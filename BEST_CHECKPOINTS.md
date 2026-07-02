# Released Checkpoints

Large checkpoint and pretrained weight files are tracked with Git LFS in this repository. The files below are included or intentionally documented for this release.

## DFormer / CoLA

- `DFormer/checkpoints/NYUD_CD/epoch-292_miou_55.05.pth` (verified mIoU=55.14 on NYUDepthv2)
- `DFormer/checkpoints/NYUD_MD/epoch-289_miou_54.94.pth` (recorded mIoU=54.94)
- `DFormer/checkpoints/NYUDepthv2_DFormer-Base_20251202-002616/epoch-150_miou_44.56.pth` (recorded mIoU=44.56)
- `DFormer/checkpoints/NYUDepthv2_DFormer-Base_20251201-215927/epoch-1_miou_21.6.pth` (recorded mIoU=21.6)

## Sigma_CD

- Local trained Sigma checkpoints were not present in the CoLA workspace.
- VMamba pretrained weights used by the Sigma variants are included when the actual local weight file is available:
  - `Sigma_CD/pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth`

Note: the `vssmbase` and `vssmtiny` entries in this workspace are incomplete placeholder pointer files rather than actual local weight files, so they are intentionally not released here.

Generated from source workspace: `E:\code\CoLA`
