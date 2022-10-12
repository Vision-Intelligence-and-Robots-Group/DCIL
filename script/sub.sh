#Cosine Normalization (Mimic Scores) + Less-forget Constraint (Adaptive Loss Weight) + Margin Ranking Loss
GPU=2
GRAPH_LAMDA=15
NBCL=10
NB_PROTOS=20
PREFIX=subimagenet

export CUDA_VISIBLE_DEVICES=$GPU 
python ./DCID_subimagenet.py\
    --nb_cl_fg 50 --nb_cl $NBCL --nb_protos $NB_PROTOS \
    --resume --imprint_weights  --epochs 20\
    --lamda 5 --graph_lambda $GRAPH_LAMDA --ref_nn 1 --cls_weight 1\
    --random_seed 2022 --round 2 --pubdataset 20\
    --dist 0.5 --K 2 --lw_mr 1 --users 5 --base_epochs 110 --log_dir './log/2022' --cil_method 'lucir'\
    --ckp_prefix $PREFIX
