#!/usr/bin/sh


corrupt_data () {
    ./inpaint.py --output $1-rnd13.h5  --corruptor rnd13  $2 $3 $4 $5
    ./inpaint.py --output $1-rnd15.h5  --corruptor rnd15  $2 $3 $4 $5
    ./inpaint.py --output $1-rnd110.h5 --corruptor rnd110 $2 $3 $4 $5
    ./inpaint.py --output $1-10x10.h5  --corruptor 10x10  $2 $3 $4 $5
    ./inpaint.py --output $1-12x12.h5  --corruptor 12x12  $2 $3 $4 $5
    ./inpaint.py --output $1-14x14.h5  --corruptor 14x14  $2 $3 $4 $5
}

#corrupt_data recons-200   output/bs25-lr13-si2-spl10-sbn-sbn-200.2014-11-10-14-53/ sbn sbn 200
#corrupt_data recons-1100  output/bs25-lr13-si2-spl10-sbn-sbn-100-100.2014-11-12-14-16/ sbn sbn 100,100
#corrupt_data recons-2100  output/bs25-lr13-si2-spl10-sbn-sbn-200-100.2014-11-10-12-28/ sbn sbn 200,100
#corrupt_data recons-4100  output/bs25-lr13-si2-spl10-sbn-sbn-400-100.2014-11-12-14-16/ sbn sbn 400,100
#corrupt_data recons-8100  output/bs100-lr13-si2-spl10-sbn-sbn-800-100.2014-11-14-13-37/ sbn sbn  800,100
corrupt_data recons-42100 output/bs100-lr13-si2-spl10-sbn-sbn-400-200-100.2014-11-14-13-37 sbn sbn 400,200,100
#corrupt_data recons-22100 output/bs25-lr13-si2-spl10-sbn-sbn-200-200-100.2014-11-08-23-33/ sbn sbn 200,200,100
#corrupt_data recons-42100            sbn sbn 400,200,100
#corrupt_data recons-32210 ~/LISA/reweighted-ws/mnist/output/bs25-lr13-si2-spl10-sbn-sbn-300-200-200-100.2014-11-05-12-58/  sbn sbn 300,200,200,100

