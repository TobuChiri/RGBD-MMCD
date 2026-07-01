# NYU
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0" -n "nyu" -e="/home/um202276680/JJW/Sigma_CD/log_final/log_nyudepth/log_NYUDepthv2_sigma_small_cromb_conmb_cvssdecoder/checkpoint/epoch-300.pth"
# SUN
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0" -n "sun" -e="/home/um202276680/JJW/Sigma_CD/log_final/log_sunrgbd/log_SUNRGBD_sigma_small_cromb_conmb_cvssdecoder/checkpoint/epoch-200.pth"