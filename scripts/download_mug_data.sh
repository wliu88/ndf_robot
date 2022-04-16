if [ -z $NDF_SOURCE_DIR ]; then echo 'Please source "ndf_env.sh" first'
else
wget -O /home/weiyu/data_drive/ndf_data/ndf_mug_data.tar.gz https://www.dropbox.com/s/42owfein4jtobd5/ndf_mug_data.tar.gz?dl=0 --no-check-certificate
wget -O /home/weiyu/data_drive/ndf_data/ndf_occ_data.tar.gz https://www.dropbox.com/s/ok4fb045z7v8cpp/ndf_occ_data.tar.gz?dl=0 --no-check-certificate
mkdir -p /home/weiyu/data_drive/ndf_data/data/training_data
mv ndf_*_data.tar.gz /home/weiyu/data_drive/ndf_data/data/training_data
cd /home/weiyu/data_drive/ndf_data/data/training_data
tar -xzf ndf_mug_data.tar.gz
tar -xzf ndf_occ_data.tar.gz
rm ndf_*_data.tar.gz
echo "Training data for NDF mugs copied to $NDF_SOURCE_DIR/data/training_data"
fi
