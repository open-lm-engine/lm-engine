INSTANCE=i-071af86fbcca0e3f1
DISK=vol-080a22b04e85ab34b

aws ec2 attach-volume \
    --volume-id $DISK \
    --instance-id $INSTANCE \
    --device /dev/sdf
