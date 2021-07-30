SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)"/db"
LAYER="resnet50_layer43"

echo "working for $LAYER"

cd result/$LAYER"_512"
timeloop-mapper $SHELL_FOLDER/arch/components/* $SHELL_FOLDER/arch/*.yaml \
    $SHELL_FOLDER/constraints/* $SHELL_FOLDER/mapper/* $SHELL_FOLDER/resnet50/$LAYER".yaml"