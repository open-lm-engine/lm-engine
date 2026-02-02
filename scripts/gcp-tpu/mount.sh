# for formatting
# sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme0n2

sudo mount -o discard,defaults /dev/nvme0n2 ./scratch
sudo chmod a+w ./scratch
