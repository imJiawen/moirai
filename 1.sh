# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH="~/Blob_WestJP/v-jiawezhang/TSFM/sandbox478/moirai/src:$PYTHONPATH"
export PYTHONPATH="/home/v-zhangjiaw/code/moirai/src:$PYTHONPATH"
export HYDRA_FULL_ERROR=1

python -m cli.train \
  -cp conf/pretrain \
  run_name=elastst_test \
  model=elastst \
  data=lotsa_v1_unweighted
  # data=lotsa_v1_weighted


# python -m cli.train \
#   -cp conf/pretrain \
#   run_name=moirai_test \
#   model=moirai_small \
#   data=largest

