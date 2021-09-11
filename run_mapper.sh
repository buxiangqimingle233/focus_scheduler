SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)"/db"
LAYER="mnasnet_layer5"

echo "working for $LAYER"

cd result-512g/$LAYER"_2"
timeloop-mapper $SHELL_FOLDER/arch/components/* modified_arch.yaml \
    $SHELL_FOLDER/constraints/* $SHELL_FOLDER/mapper/* $SHELL_FOLDER/mnasnet/$LAYER".yaml"