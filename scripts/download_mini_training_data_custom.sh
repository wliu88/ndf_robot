# mini dataset, just to get everything up and running
wget -O /home/weiyu/data_drive/ndf_data/ndf_mug_data.tar.gz https://www.dropbox.com/s/b2xf65x1t6pgvsp/ndf_test_mug_data.tar.gz?dl=0 --no-check-certificate
wget -O /home/weiyu/data_drive/ndf_data/ndf_bottle_data.tar.gz https://www.dropbox.com/s/by38ryktxkcqxcx/ndf_test_bottle_data.tar.gz?dl=0 --no-check-certificate
wget -O /home/weiyu/data_drive/ndf_data/ndf_bowl_data.tar.gz https://www.dropbox.com/s/hd0f6deodll3z47/ndf_test_bowl_data.tar.gz?dl=0 --no-check-certificate
wget -O /home/weiyu/data_drive/ndf_data/ndf_occ_data.tar.gz https://www.dropbox.com/s/ok4fb045z7v8cpp/ndf_occ_data.tar.gz?dl=0 --no-check-certificate
mkdir -p /home/weiyu/data_drive/ndf_data/data/training_data
mv /home/weiyu/data_drive/ndf_data/ndf_*_data.tar.gz /home/weiyu/data_drive/ndf_data/data/training_data
cd /home/weiyu/data_drive/ndf_data/data/training_data
tar -xzf ndf_mug_data.tar.gz
tar -xzf ndf_bottle_data.tar.gz
tar -xzf ndf_bowl_data.tar.gz
tar -xzf ndf_occ_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data NDF copied to $NDF_SOURCE_DIR/data/training_data"
